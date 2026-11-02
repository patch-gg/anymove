#!/bin/bash
# Per creature, last step: one GLB with the textured, rigged creature and all its moves, ready for a game.
# tools/pack.sh CREATURE [picks=picks/CREATURE.json] [out=runs/CREATURE/pack]
#   -> out/CREATURE.glb, out/CREATURE_flagged.glb (picks marked "flagged"), out/clips/, shape_report.json, locomotion.json
# Trims one-shots, cuts seamless loops, makes root-motion versions of walks and runs, and takes the limb paint out.
# Needs assets/CREATURE/rigged/rigged.blend (tools/rig.sh) and each pick's runs/CREATURE/clips/<take>/motion.json
# (tools/capture.sh). No GPU. See docs/PACKS.md.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
c=$1; picks=${2:-picks/$c.json}; out=${3:-runs/$c/pack}; rig=assets/$c/rigged
[ -f "$picks" ] || { echo "no $picks: list the takes to pack (docs/PACKS.md)" >&2; exit 1; }
$PY tools/shape_pack.py "$c" "$picks" "$out"
rm -f "${out:?}/${c:?}.glb" "${out:?}/${c:?}_flagged.glb"
while read -r glb clips; do
  # one Name=path per word, and none holds a space: they're relative to the repo root
  run_blender -b -P blender/pack_glb.py -- "$rig/rigged.blend" "$rig/rig_info.json" "$out/$glb" $clips \
    | grep -E "^(clip|glb written)|Error" || true
  [ -f "$out/$glb" ] || { echo "Blender wrote no $out/$glb" >&2; exit 1; }
done < "$out/pack_args.txt"
$PY tools/locomotion.py "$c" "$picks" "$out"
echo "done: $out/$c.glb"
