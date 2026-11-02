#!/bin/bash
# Once per creature, step 1: scale the mesh to real size and stand it on the ground, then draw side, front and top
# views with a 10 cm grid (and rig.json's joints, if there is one) for placing the skeleton.
# tools/prep.sh CREATURE      reads assets/CREATURE/model.glb and creature.json "prep": length_m, rotz_deg, axis
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
a=assets/$1
read -r len rot axis < <(pyout -c "import json; p = json.load(open('$a/creature.json')).get('prep', {}); print(p.get('length_m', 1.0), p.get('rotz_deg', 0), p.get('axis', 'y'))")
run_blender -b -P blender/prep_mesh.py -- "$a/model.glb" "$len" "$a/prep" "$rot" "$axis" | grep -E "^size|^saved|Error" || true
run_blender -b -P blender/ortho_views.py -- "$a/prep/static.blend" "$a/prep/views" 300 | grep -E "^views|Error" || true
rig=$a/rig.json; [ -f "$rig" ] || rig=-
$PY tools/grid_views.py "$a/prep/views" "$a/prep/views.png" "$rig"
echo "check $a/prep/views.png: the creature faces -Y (head to the right in the side view), feet on z = 0"
