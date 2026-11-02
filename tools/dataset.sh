#!/bin/bash
# Once per creature, step 3: render COUNT random poses inside the joint limits from both cameras, with the true pixel
# position of every joint, in PROCS Blender processes; then a contact sheet of the first samples.
# tools/dataset.sh CREATURE [count=2200] [procs=3] [seed=500]     -> runs/CREATURE/dataset
# About 25 min for 2,200 poses on one RTX 4090. Don't run more than 2 Blender processes beside a reader training.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
a=assets/$1; n=${2:-2200}; p=${3:-3}; seed=${4:-500}
per=$(( (n + p - 1) / p ))
mkdir -p "runs/$1/dataset"
for k in $(seq 0 $((p - 1))); do
  run_blender -b -P blender/render_dataset.py -- "$a/rigged/rigged.blend" "$a/rigged/rig_info.json" "runs/$1/dataset" \
    $((k * per)) "$per" "$seed" > "runs/$1/dataset_$k.log" 2>&1 &
done
wait
$PY tools/show_samples.py "runs/$1/dataset" "runs/$1/dataset_samples.jpg" 8
echo "$(ls "runs/$1/dataset" | grep -c _meta.json) samples; true joints drawn in runs/$1/dataset_samples.jpg"
