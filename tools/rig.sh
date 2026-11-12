#!/bin/bash
# Once per creature, step 2: build the armature from rig.json, skin it with Blender's automatic weights, paint each
# limb its colour, and render the two-camera reference frame the video model starts from.
# tools/rig.sh CREATURE preview   only the joints over an x-ray of the mesh (side, front, top), to check placement
# tools/rig.sh CREATURE           assets/CREATURE/rigged/{rigged.blend,rig_info.json}, runs/CREATURE/ref/ref.png
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
a=assets/$1
if [ "${2:-}" = preview ]; then
  run_blender -b -P blender/build_rig.py -- "$a/prep/static.blend" "$a/rig.json" "$a/prep/preview" preview > /dev/null
  echo "joints over the mesh: $a/prep/preview/joints_side.png, joints_front.png, joints_top.png"
  exit 0
fi
run_blender -b -P blender/build_rig.py -- "$a/prep/static.blend" "$a/rig.json" "$a/rigged" \
  | grep -E "merged|without weights|smoothed|finger weights|^prop |closed around|painted|saved|Error" || true
run_blender -b -P blender/render_ref.py -- "$a/rigged/rigged.blend" "runs/$1/ref" | grep -E "Error" || true
$PY tools/compose_ref.py "runs/$1/ref"
echo "test pose: $a/rigged/testpose_front.png, testpose_side.png"
