#!/usr/bin/env bash
# Prepare the offline wheelhouse runpod/wheels/ — the only networked step of option A.
# Runs inside the project's base image (exact Python/platform as the final image):
#   1. downloads binary wheels for runpod/requirements.txt and all their dependencies;
#   2. builds wheels for local packages listed in runpod/build-contexts.conf.
# Afterwards ./runpod/build.sh installs only from runpod/wheels/ with --network=none.
# Rerun only when requirements.txt, a local package or the base image changes.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: ./runpod/wheels.sh'
  echo 'Download linux wheels for runpod/requirements.txt inside the base image and build local'
  echo 'package wheels from build-contexts.conf into runpod/wheels/. Needs network once.'
  echo 'Set PIP_INDEX_URL / PIP_EXTRA_INDEX_URL in the environment to use another index.'
  exit 0
fi
runpod_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(dirname "$runpod_dir")"
die() { echo "wheels.sh: $*" >&2; exit 2; }

base="$(sed -n 's/^ARG BASE_IMAGE=//p' "$runpod_dir/Dockerfile" | head -n 1)"
[[ -n "$base" ]] || die 'runpod/Dockerfile must declare ARG BASE_IMAGE=<image> before FROM ${BASE_IMAGE}'
[[ -f "$runpod_dir/requirements.txt" ]] || die 'missing runpod/requirements.txt'
docker --context orbstack image inspect "$base" >/dev/null 2>&1 \
  || die "base image $base not in local OrbStack; pull it first (see shared README)"

staging="$runpod_dir/.wheels.tmp"
rm -rf "$staging"
mkdir -p "$staging"
trap 'rm -rf "$staging"' EXIT

mounts=(-v "$runpod_dir/requirements.txt:/in/requirements.txt:ro" -v "$staging:/out")
script='set -euo pipefail
pip download --no-cache-dir --only-binary=:all: --dest /out -r /in/requirements.txt'
if [[ -f "$runpod_dir/build-contexts.conf" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"; line="${line//[[:space:]]/}"
    [[ -z "$line" ]] && continue
    name="${line%%=*}"; path="${line#*=}"
    [[ -n "$name" && -n "$path" && "$line" == *=* ]] || die "invalid build-contexts.conf line: $line"
    abs="$(cd "$root/$path" 2>/dev/null && pwd -P)" || die "context $name not found: $root/$path"
    [[ -f "$abs/pyproject.toml" ]] || die "context $name has no pyproject.toml: $abs"
    mounts+=(-v "$abs:/ctx/$name:ro")
    # Build from a copy: the mount is read-only and private files are never written back.
    script+="
cp -r /ctx/$name /tmp/$name && pip wheel --no-cache-dir --no-deps --wheel-dir /out /tmp/$name"
  done < "$runpod_dir/build-contexts.conf"
fi

echo "Base image: $base"
docker --context orbstack run --rm --platform linux/amd64 --entrypoint bash \
  -e PIP_INDEX_URL -e PIP_EXTRA_INDEX_URL "${mounts[@]}" "$base" -c "$script"

rm -rf "$runpod_dir/wheels"
mv "$staging" "$runpod_dir/wheels"
trap - EXIT
count="$(find "$runpod_dir/wheels" -name '*.whl' | wc -l | tr -d ' ')"
echo "Prepared $count wheels in runpod/wheels/ ($(du -sh "$runpod_dir/wheels" | cut -f1)). Build offline: ./runpod/build.sh"
