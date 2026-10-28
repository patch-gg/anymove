#!/bin/bash
# Once per creature: make model.glb from a concept image, Rodin first, then fal Tripo if generation fails.
# tools/mesh.sh CREATURE IMAGE [--resume]     -> assets/CREATURE/model.glb, runs/CREATURE/rodin*.json
# Rodin defaults to the public vibecoding key; fal fallback needs FAL_KEY and is billed by fal.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
c=$1; image=${2:?concept image}
shift 2
$PY -m mocap.mesh "$image" "assets/$c/model.glb" "runs/$c/rodin.json" "$@"
