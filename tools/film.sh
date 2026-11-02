#!/bin/bash
# Per move: send the reference frame and prompts/CREATURE/MOVE.txt to video models on fal, one clip per seed.
# tools/film.sh CREATURE MOVE [models=h3max] [seeds=1,2]     -> runs/CREATURE/videos/MOVE/s<seed>/<model>.mp4
# Needs FAL_KEY. MiniMax H3 Max is about $0.26 a 5 s clip (October 2026), and a usable move took about 7 clips on
# average; models: h3, h3max, h3maxturbo, seedance, flux3max.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
c=$1; m=${2:?move}; models=${3:-h3max}; seeds=${4:-1,2}
[ -f "prompts/$c/$m.txt" ] || { echo "write prompts/$c/$m.txt first (tools/make_prompt.py)" >&2; exit 1; }
for s in ${seeds//,/ }; do
  $PY -m mocap.generate "runs/$c/ref/ref.png" "prompts/$c/$m.txt" "runs/$c/videos/$m/s$s" "$models" 5 "$s" &
done
wait
ls runs/$c/videos/$m/s*/*.mp4
