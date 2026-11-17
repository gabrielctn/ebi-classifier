#!/bin/bash
# Start the complete, offline EBI Classifier from its extracted folder.
set -u
cd "$(dirname "$0")" || exit 1
script="$PWD/START_EBI.sh"

# When a file manager executes a shell script without a terminal, open one so
# the user can see progress, errors, and close it to stop the application.
if [ ! -t 0 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ "${EBI_IN_TERMINAL:-}" != 1 ]; then
    export EBI_IN_TERMINAL=1
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal -- /bin/bash "$script"; exit $?
    elif command -v konsole >/dev/null 2>&1; then
        konsole -e /bin/bash "$script"; exit $?
    elif command -v mate-terminal >/dev/null 2>&1; then
        mate-terminal -- /bin/bash "$script"; exit $?
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal -x /bin/bash "$script"; exit $?
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        x-terminal-emulator -e /bin/bash "$script"; exit $?
    elif command -v xterm >/dev/null 2>&1; then
        xterm -e /bin/bash "$script"; exit $?
    fi
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --text="Open a terminal in the extracted folder and run ./START_EBI.sh"
    fi
    exit 1
fi

echo
echo '==============================================='
echo '              EBI Classifier'
echo '==============================================='
echo

fail() {
    echo
    echo "$1"
    echo 'Please send a photo of this entire window to the person who sent you the app.'
    if [ -t 0 ]; then
        echo 'Press Return to close this window.'
        read -r _
    fi
    exit 1
}

[ "$(uname -s)" = Linux ] || fail 'This package is for Linux. Use the Mac or Windows package on those systems.'
case "$(uname -m)" in
    x86_64|amd64) platform=x86_64 ;;
    aarch64|arm64) platform=aarch64 ;;
    *) fail 'This Linux processor is not supported. Use a 64-bit Intel/AMD or ARM computer.' ;;
esac

libc="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
case "$libc" in
    'glibc '*) version="${libc#glibc }" ;;
    *) fail 'This package requires a Linux distribution using glibc 2.28 or newer.' ;;
esac
IFS=. read -r libc_major libc_minor _ <<< "$version"
case "${libc_major:-}${libc_minor:-}" in
    ''|*[!0-9]*) fail 'Could not determine the Linux system library version.' ;;
esac
if [ "$libc_major" -lt 2 ] || { [ "$libc_major" -eq 2 ] && [ "$libc_minor" -lt 28 ]; }; then
    fail 'This Linux system is too old. The package requires glibc 2.28 or newer.'
fi

runtime="$PWD/runtime/$platform"
envdir="$PWD/.venv-$platform"
uv="$runtime/uv"
python="$runtime/python/bin/python3.14"
requirements="$PWD/linux/requirements.txt"
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
