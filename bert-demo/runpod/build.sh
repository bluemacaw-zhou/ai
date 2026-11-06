#!/usr/bin/env bash
# Portable project entrypoint; canonical copy: ~/workspace/runpod/templates/project-build.sh
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: ./runpod/build.sh [--version TAG] [--local-only] [--tag-only] [--push] [--dry-run]
Default: infer project root/name, build linux/amd64, load locally, add GHCR tag.
--version TAG  Override runpod/image.conf version (default: 0.1.0).
--local-only   Build/tag locally without requiring GITHUB_USER.
--tag-only     Reuse the existing local image instead of rebuilding.
--push         Explicitly upload the resulting GHCR image (requires prior docker login).
--dry-run      Print actions without contacting Docker or changing files.
EOF
}
version_override='' local_only=false tag_only=false push=false dry_run=false
while (( $# )); do
  case "$1" in
    --version) [[ $# -ge 2 ]] || { echo 'Missing --version value' >&2; exit 2; }; version_override="$2"; shift 2 ;;
    --local-only) local_only=true; shift ;;
    --tag-only) tag_only=true; shift ;;
    --push) push=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
fail() { echo "Error: $*" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ -f "$root/runpod/Dockerfile" ]] || fail 'Place this script at PROJECT/runpod/build.sh with a Dockerfile alongside it.'
project="$(basename "$root" | tr '[:upper:]' '[:lower:]')"
version='0.1.0'
if [[ -f "$root/runpod/image.conf" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    case "$line" in
      version=*) version="${line#version=}" ;;
      *) fail 'image.conf supports only version=TAG (no shell expressions).' ;;
    esac
  done < "$root/runpod/image.conf"
fi
[[ -z "$version_override" ]] || version="$version_override"
[[ "$project" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] || fail 'Project directory must be a valid lowercase image name.'
[[ "$version" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]] || fail 'Invalid image version.'
local_image="$project:$version"
image="$local_image"
if ! "$local_only"; then
  [[ -n "${GITHUB_USER:-}" ]] || fail 'GITHUB_USER is missing. Run source ~/.zshrc, or use --local-only.'
  namespace="$(printf '%s' "$GITHUB_USER" | tr '[:upper:]' '[:lower:]')"
  [[ "$namespace" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || fail 'GITHUB_USER must be a GitHub username.'
  image="ghcr.io/$namespace/$project:$version"
fi
if "$local_only" && "$push"; then fail '--push requires a GHCR tag; remove --local-only.'; fi
# Offline builds install from runpod/wheels/; fail early instead of letting Docker fail midway.
if ! "$tag_only" && grep -q 'runpod/wheels' "$root/runpod/Dockerfile" \
   && ! compgen -G "$root/runpod/wheels/*.whl" >/dev/null; then
  fail 'runpod/wheels/ is empty. Prepare the offline wheelhouse first: ./runpod/wheels.sh'
fi
build=(docker buildx build --platform linux/amd64 --file "$root/runpod/Dockerfile" --tag "$local_image" --progress plain --load)
if ! "$tag_only" && [[ -f "$root/runpod/build-contexts.conf" ]]; then
  names=' '
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || fail 'Build context must be NAME=PATH.'
    name="${line%%=*}"; path="${line#*=}"
    [[ "$name" =~ ^[a-zA-Z][a-zA-Z0-9_-]*$ && -n "$path" ]] || fail 'Invalid build context.'
    [[ "$names" != *" $name "* ]] || fail "Duplicate build context: $name"
    names="$names$name "
    [[ "$path" == /* ]] || path="$root/$path"
    [[ -d "$path" ]] || fail "Missing build context directory: $path"
    path="$(cd "$path" && pwd -P)"
    build+=(--build-context "$name=$path")
  done < "$root/runpod/build-contexts.conf"
fi
build+=("$root")
run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if ! "$dry_run"; then "$@"; fi
}
printf 'Project: %s\nLocal image: %s\nTarget image: %s\n' "$root" "$local_image" "$image"
if "$tag_only"; then
  run docker image inspect "$local_image" --format '{{.Id}}'
else
  run "${build[@]}"
fi
if ! "$local_only"; then run docker tag "$local_image" "$image"; fi
if ! "$dry_run"; then
  ref_tmp="$(mktemp "$root/runpod/.image-ref.XXXXXX")"
  printf '%s\n' "$image" > "$ref_tmp"
  mv "$ref_tmp" "$root/runpod/.image-ref"
fi
if "$push"; then run docker push "$image"; fi
printf 'Image: %s\n' "$image"
if ! "$push"; then echo 'Not uploaded. Use --tag-only --push when ready to publish.'; fi
