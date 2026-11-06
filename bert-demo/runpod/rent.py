#!/usr/bin/env python3
"""Use the official runpodctl CLI to select and rent one currently affordable GPU."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone


class CliError(RuntimeError):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(str(payload.get("error", "Runpod CLI request failed")))


def cli(*args, timeout=45):
    # Never retry a mutation here. A timeout may leave a real billable Pod behind.
    try:
        result = subprocess.run(["runpodctl", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CliError({"code": "network_error", "error": "CLI timed out; the request outcome may be unknown"}) from exc
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise CliError({"code": "invalid_response", "error": "CLI returned no valid JSON; inspect runpodctl directly"}) from exc
    if result.returncode or (isinstance(payload, dict) and "error" in payload):
        if not isinstance(payload, dict):
            payload = {"code": "cli_error", "error": "CLI failed with an unexpected response"}
        raise CliError(payload)
    return payload


def save(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read JSON configuration/state: " + str(path)) from exc


def config_for(directory, args):
    config = read_json(directory / "rent.json")
    if not isinstance(config, dict):
        raise RuntimeError("rent.json must contain an object")
    if args.max_price is not None:
        config["max_gpu_price_per_hour"] = args.max_price
    if args.min_vram is not None:
        config["min_vram_gb"] = args.min_vram
    if config.get("cloud_type") not in ("COMMUNITY", "SECURE"):
        raise RuntimeError("cloud_type must be COMMUNITY or SECURE")
    for name in ("max_gpu_price_per_hour", "min_vram_gb"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(name + " must be a positive finite number")
    for name in ("container_disk_gb", "volume_gb", "max_capacity_attempts"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(name + " must be a positive integer")
    if config["max_capacity_attempts"] > 10:
        raise RuntimeError("max_capacity_attempts must be at most 10")
    if not re.fullmatch(r"\d+\.\d+", str(config.get("min_cuda_version", ""))):
        raise RuntimeError("Set an explicit min_cuda_version, e.g. 12.8")
    excluded = config.get("excluded_gpu_ids", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) or not item for item in excluded):
        raise RuntimeError("excluded_gpu_ids must be a list of GPU type names")
    re.compile(config["gpu_name_pattern"])
    config["image_override"] = getattr(args, "image", None)
    config["public_image"] = getattr(args, "public_image", False)
    if config["public_image"] and not config["image_override"]:
        raise RuntimeError("--public-image requires an explicit --image")
    if config["image_override"] and not re.fullmatch(r"[a-z0-9][a-z0-9._:/-]*(?::[A-Za-z0-9_.-]+|@sha256:[a-f0-9]{64})", config["image_override"]):
        raise RuntimeError("--image requires an explicit version tag or sha256 digest")
    return config


def candidates(catalog, config, excluded=()):
    if not isinstance(catalog, list):
        raise RuntimeError("gpu list returned an unexpected structure")
    cloud = "community" if config["cloud_type"] == "COMMUNITY" else "secure"
    pattern = re.compile(config["gpu_name_pattern"])
    output = []
    for gpu in catalog:
        gpu_id = gpu.get("gpuId", "")
        price = gpu.get(cloud + "PricePerHr")
        memory = gpu.get("memoryInGb", 0)
        if (gpu.get("available") is not True or gpu.get(cloud + "Cloud") is not True
                or gpu_id in excluded or gpu_id in config.get("excluded_gpu_ids", []) or not gpu_id.startswith("NVIDIA ") or not pattern.fullmatch(gpu_id)
                or isinstance(price, bool) or not isinstance(price, (int, float))
                or not math.isfinite(price) or price <= 0 or price > config["max_gpu_price_per_hour"]
                or not isinstance(memory, (int, float)) or memory < config["min_vram_gb"]):
            continue
        output.append({"gpu_id": gpu_id, "price": price, "memory": memory,
                       "stock": gpu.get("stockStatus", "unknown")})
    return sorted(output, key=lambda item: (item["price"], -item["memory"], item["gpu_id"]))


def show(items, config):
    print("Cloud: {} | GPU quote limit: ${:.3f}/h | minimum VRAM: {} GB".format(
        config["cloud_type"], config["max_gpu_price_per_hour"], config["min_vram_gb"]), flush=True)
    print("GPU hourly quotes exclude disk charges; this is not a total-spend or server-side price lock.", flush=True)
    for gpu in items[:10]:
        print("  ${:.3f}/h  {:g} GB  {}  stock={}".format(
            gpu["price"], gpu["memory"], gpu["gpu_id"], gpu["stock"]), flush=True)


def pods_named(name):
    pods = cli("pod", "list", "--all")
    if not isinstance(pods, list):
        raise RuntimeError("Unexpected Pod list; refusing another creation")
    return [pod for pod in pods if pod.get("name") == name]


def handoff(pod_id):
    print("Pod ID: " + pod_id, flush=True)
    print("The Pod may still be pulling/booting. No additional Pod was requested.")
    print("Inspect: runpodctl pod get " + pod_id)
    print("Logs:    runpodctl pod logs " + pod_id + " --source system")
    print("SSH:     runpodctl ssh info " + pod_id)
    print("Stop:    runpodctl pod stop " + pod_id)
    print("Checkpoints: /workspace. Download before deleting this Pod; its volume disk is Pod-bound.")


def existing_state(state_path):
    if not state_path.exists():
        return False
    state = read_json(state_path)
    if state.get("pod_id"):
        print("A previous Pod is already recorded; refusing to rent another.")
        handoff(str(state["pod_id"]))
        return True
    if state.get("status") in ("requesting", "uncertain"):
        matches = pods_named(state["name"])
        if len(matches) == 1 and matches[0].get("id"):
            state.update(status="created", pod_id=matches[0]["id"])
            save(state_path, state)
            handoff(state["pod_id"])
            return True
        raise RuntimeError("Previous creation outcome is unresolved. Inspect runpodctl pod list --all; "
                           "do not create again until absence is confirmed. State: " + str(state_path))
    if state.get("status") == "rejected":
        return False  # Explicit capacity rejection was already reconciled with the account list.
    raise RuntimeError("Unknown rental state; inspect " + str(state_path))


def preflight(directory, config):
    image = config.get("image_override") or (directory / ".image-ref").read_text().strip()
    if not config.get("image_override") and not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+(?::[A-Za-z0-9_.-]+|@sha256:[a-f0-9]{64})", image):
        raise RuntimeError("Build/tag first: runpod/.image-ref must contain a versioned GHCR image")
    key_path = Path(config["ssh_public_key"]).expanduser()
    if not key_path.is_absolute():
        key_path = directory.parent / key_path
    key = key_path.read_text().strip()
    if not key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")) or "PRIVATE KEY" in key:
        raise RuntimeError("ssh_public_key must point to a public SSH key")
    if config.get("public_image"):
        return image, key, None
    registries = cli("registry", "list")
    if not isinstance(registries, list):
        raise RuntimeError("Unexpected registry list")
    matches = [entry for entry in registries if entry.get("name") == config["registry_auth_name"]]
    if len(matches) != 1 or not matches[0].get("id"):
        raise RuntimeError("Expected one registry credential named " + config["registry_auth_name"])
    return image, key, matches[0]["id"]


def is_capacity_rejection(payload):
    # Only explicit HTTP-400 placement failures are retryable; never retry an ambiguous timeout/5xx.
    if payload.get("code") != "bad_request" or payload.get("id"):
        return False
    message = str(payload.get("error", "")).lower()
    return any(phrase in message for phrase in (
        "no instances available", "no available instances", "no available machines",
        "no available gpus", "insufficient capacity"))


def rent(directory, config):
    state_path = directory / ".rental.json"
    if existing_state(state_path):
        return
    image, key, registry_id = preflight(directory, config)
    print("Image: " + image, flush=True)
    print("Registry authentication: " + ("none (public image)" if registry_id is None else "configured credential"), flush=True)
    project = re.sub(r"[^a-z0-9-]", "-", directory.parent.name.lower())[:32]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = project + "-" + stamp + "-" + uuid.uuid4().hex[:6]
    excluded = set()
    for _ in range(config["max_capacity_attempts"]):
        # Refresh immediately before each attempt; no credentials or file setup after this query.
        items = candidates(cli("gpu", "list"), config, excluded)
        if not items:
            raise RuntimeError("No available GPU matches the configured budget/model/VRAM limits. No Pod created.")
        selected = items[0]
        show(items, config)
        print("Creating one Pod with {} at catalog quote ${:.3f}/h...".format(
            selected["gpu_id"], selected["price"]), flush=True)
        state = {"status": "requesting", "name": name, "image": image,
                 "gpu_id": selected["gpu_id"], "quoted_gpu_price_per_hour": selected["price"],
                 "cloud_type": config["cloud_type"], "created_at": datetime.now(timezone.utc).isoformat()}
        save(state_path, state)
        try:
            pod = cli("pod", "create", "--name", name, "--image", image,
                      *(["--registry-auth-id", registry_id] if registry_id else []),
                      "--gpu-id", selected["gpu_id"],
                      "--gpu-count", "1", "--cloud-type", config["cloud_type"],
                      "--min-cuda-version", config["min_cuda_version"],
                      "--container-disk-in-gb", str(config["container_disk_gb"]),
                      "--volume-in-gb", str(config["volume_gb"]), "--volume-mount-path", "/workspace",
                      "--ports", "22/tcp", "--env", json.dumps({"SSH_PUBLIC_KEY": key, "PUBLIC_KEY": key}), timeout=90)
        except CliError as exc:
            if exc.payload.get("id"):
                state.update(status="created", pod_id=exc.payload["id"])
                save(state_path, state)
                handoff(str(state["pod_id"]))
                return
            state["status"] = "uncertain"
            save(state_path, state)
            matches = pods_named(name)  # If reconciliation fails, stop with uncertain state intact.
            if len(matches) == 1 and matches[0].get("id"):
                state.update(status="created", pod_id=matches[0]["id"])
                save(state_path, state)
                handoff(str(state["pod_id"]))
                return
            if not matches and is_capacity_rejection(exc.payload):
                state.update(status="rejected", reason="explicit capacity rejection; no matching Pod found")
                save(state_path, state)
                excluded.add(selected["gpu_id"])
                print("Capacity was taken; refreshing inventory and trying the next GPU class.", flush=True)
                continue
            raise RuntimeError("Creation failed or is uncertain; no further GPU will be requested. "
                               "Inspect runpodctl pod list --all. CLI: " + str(exc)) from exc
        if not isinstance(pod, dict) or not pod.get("id"):
            state["status"] = "uncertain"
            save(state_path, state)
            raise RuntimeError("Create returned no Pod ID. Inspect the account before retrying.")
        state.update(status="created", pod_id=pod["id"])
        save(state_path, state)
        handoff(str(pod["id"]))
        return
    raise RuntimeError("Maximum capacity attempts reached; no confirmed Pod was created.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="explicitly rent one billable Pod; no interactive prompt")
    parser.add_argument("--max-price", type=float, help="override GPU quote ceiling in USD/hour, excluding storage")
    parser.add_argument("--min-vram", type=float, help="override minimum GPU memory in GB")
    parser.add_argument("--image", help="override .image-ref with a versioned container image")
    parser.add_argument("--public-image", action="store_true", help="omit registry credentials; requires --image")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    config = config_for(directory, args)
    if not args.create:
        items = candidates(cli("gpu", "list"), config)
        show(items, config)
        if not items:
            raise RuntimeError("No GPU matches these limits")
        print("Preview only. Run ./runpod/rent.sh --create to refresh stock and rent immediately.")
        return
    # macOS/Linux advisory lock: prevent concurrent invocations from creating duplicates.
    with (directory / ".rent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another rental command is in progress") from exc
        rent(directory, config)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, re.error) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
