# Shared by the tools/*.sh scripts: run from the repo root, find Python and Blender. Works in Git Bash on Windows.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe  # a venv on Windows
elif command -v python3 > /dev/null; then PY=python3
else PY=python
fi
export PYTHONUTF8=1  # creature.json and prompts are UTF-8; Python on Windows would otherwise read them as cp1252
BLENDER=${BLENDER:-blender}
need_creature() {
  [ -n "${1:-}" ] && [ -f "assets/$1/creature.json" ] || { echo "no assets/${1:-<creature>}/creature.json" >&2; exit 1; }
  export MOCAP_CREATURE=assets/$1
}
# Python, for output read by `read` or $(...): on Windows it ends lines with \r\n, and `read` would keep the \r.
pyout() { $PY "$@" | tr -d '\r'; }
# Blender, without the add-ons of whoever runs it. On Windows it resolves a relative output path against the drive root
# (C:\assets\...), not the working directory, so the script's path arguments (those after --) go in absolute. Name=path
# arguments are read by Python, which gets relative paths right.
run_blender() {
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
      local args=() after=0 a
      for a in "$@"; do
        if [ $after = 1 ] && [[ $a == */* ]] && [[ $a != -* ]] && [[ $a != *=* ]] && [[ $a != /* ]] && [[ ! $a =~ ^[A-Za-z]: ]]; then
          a=$(cygpath -m "$PWD/$a")
        fi
        [ "$a" = "--" ] && after=1
        args+=("$a")
      done
      "$BLENDER" --factory-startup "${args[@]}"
      ;;
    *) "$BLENDER" --factory-startup "$@" ;;
  esac
}
