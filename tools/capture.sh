#!/bin/bash
# Per clip: read every joint in both views, fit the rig, and write the animation as a GLB with renders and a
# side-by-side comparison of the AI clip and the rig.
# tools/capture.sh CREATURE VIDEO.mp4 NAME [views=persp,side]     -> runs/CREATURE/clips/NAME/
#   det.npz (joints read), overlay.mp4, motion.json (the fit), render/animation.glb, showcase.mp4
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
c=$1; video=${2:?video}; name=${3:?name}; views=${4:-persp,side}
rig=assets/$c/rigged; out=runs/$c/clips/$name
mkdir -p "$out"
$PY -m mocap.detect "$video" "runs/$c/reader.pt" "runs/$c/ref/ref.png" "$out/det.npz" "$out/overlay.mp4" | tail -1
$PY -m mocap.solve "$out/det.npz" "$rig/rig_info.json" "$out/motion.json" | grep -v "iter"
# the 3/4 showcase camera: creature.json "camera", or one that frames this take (tools/frame_camera.py)
read -r cam target lens < <(pyout tools/frame_camera.py --creature "assets/$c/creature.json" "$out/motion.json")
run_blender -b -P blender/apply_motion.py -- "$rig/rigged.blend" "$rig/rig_info.json" "$out/motion.json" "$out/render" \
  --views "$views" --samples 24 --cam "$cam" --target "$target" --lens "$lens" --glb > "$out/blender.log" 2>&1
$PY -m mocap.showcase "$video" "$out/det.npz" "$out/render" "$out/showcase.mp4" "$c $name"
if command -v ffmpeg > /dev/null; then
  ffmpeg -loglevel error -y -i "$out/showcase.mp4" -c:v libx264 -crf 20 -pix_fmt yuv420p "$out/showcase_h264.mp4"
fi
echo "done: $out (animation: $out/render/animation.glb)"
