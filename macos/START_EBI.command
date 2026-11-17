#!/bin/bash
# Double-click this file after extracting the entire EBI Classifier folder.
set -u
cd "$(dirname "$0")" || exit 1

echo
echo '==============================================='
echo '              EBI Classifier'
echo '==============================================='
echo

fail() {
    echo
    echo "$1"
    echo 'Please send a photo of this entire window to the person who sent you the app.'
    echo 'Press Return to close this window.'
    read -r _
    exit 1
}

case "$(uname -m)" in
    arm64) platform=arm64; python_version=3.14 ;;
    x86_64) platform=intel; python_version=3.11 ;;
    *) fail 'This Mac processor is not supported.' ;;
esac

runtime="$PWD/runtime/$platform"
envdir="$PWD/.venv-$platform"
uv="$runtime/uv"
python="$runtime/python/bin/python$python_version"
requirements="$PWD/macos/requirements-$platform.txt"

[ -x "$uv" ] && [ -x "$python" ] && [ -d "$PWD/runtime/wheels" ] && \
    [ -f "$requirements" ] && [ -f "$PWD/runs/release/final.safetensors" ] && \
    [ -f "$PWD/runs/release/final.json" ] || \
    fail 'Some application files are missing. Extract the complete ZIP and keep the files together.'

export UV_NO_CONFIG=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export PYTHONUTF8=1

if [ ! -x "$envdir/bin/python" ] || ! "$envdir/bin/python" -c 'pass' >/dev/null 2>&1; then
    echo 'Preparing the application for this computer...'
    echo 'This can take a few minutes the first time.'
    echo
    "$uv" venv --offline --python "$python" --clear "$envdir" || \
        fail 'The app could not be prepared. Check free space and that this folder is writable.'
fi

echo 'Checking the application files. Please keep this window open...'
"$uv" pip sync --offline --no-index --find-links "$PWD/runtime/wheels" \
    --python "$envdir/bin/python" "$requirements" || \
    fail 'Setup failed. Check free space and that this folder is writable.'

echo
echo 'Starting the classifier. Your browser should open automatically.'
echo 'If it does not, open the local address shown below in your browser.'
echo 'Keep this window open while using the classifier.'
echo 'To stop, close this window.'
echo
"$envdir/bin/python" app.py || fail 'The classifier stopped with an error. See START_HERE.txt for help.'
