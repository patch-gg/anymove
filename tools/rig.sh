#!/bin/bash
# Once per creature, step 2: build the armature from rig.json, skin it with Blender's automatic weights, paint each
# limb its colour, and render the two-camera reference frame the video model starts from.
# tools/rig.sh CREATURE preview   only the joints over an x-ray of the mesh (side, front, top), to check placement
# tools/rig.sh CREATURE           assets/CREATURE/rigged/{rigged.blend,rig_info.json}, runs/CREATURE/ref/ref.png
# tools/rig.sh CREATURE rodin [--resume]   Rodin custom rigging instead of Blender automatic weights
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
a=assets/$1
if [ "${2:-}" = preview ]; then
  $BLENDER -b -P blender/build_rig.py -- "$a/prep/static.blend" "$a/rig.json" "$a/prep/preview" preview > /dev/null
  echo "joints over the mesh: $a/prep/preview/joints_side.png, joints_front.png, joints_top.png"
  exit 0
fi
if [ "${2:-}" = rodin ]; then
  c=$1; shift 2
  source_args=()
  if [ -f "runs/$c/rodin.json" ] && [ ! -f "runs/$c/rodin.tripo.json" ]; then
    asset=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["uuid"])' "runs/$c/rodin.json")
    source_args=(--asset "$asset")
  fi
  $PY -m mocap.rig "$a/model.glb" "$a/rig.json" "runs/$c/rigged.glb" "runs/$c/rigging.json" "${source_args[@]}" "$@"
  $BLENDER -b --python-exit-code 1 -P blender/import_rig.py -- "runs/$c/rigged.glb" "$a/rig.json" "$a/creature.json" "$a/rigged"
  set -- "$c"
else
  $BLENDER -b -P blender/build_rig.py -- "$a/prep/static.blend" "$a/rig.json" "$a/rigged" \
    | grep -E "merged|without weights|smoothed|painted|saved|Error" || true
fi
$BLENDER -b -P blender/render_ref.py -- "$a/rigged/rigged.blend" "runs/$1/ref" | grep -E "Error" || true
$PY tools/compose_ref.py "runs/$1/ref"
echo "test pose: $a/rigged/testpose_front.png, testpose_side.png"
