#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/dist"
mojo build --emit shared-lib "$root/src/capi.mojo" -o "$root/dist/libmojo-pyfastx.so"
suffix=$(python3-config --extension-suffix)
cc -O3 -shared -fPIC $(python3-config --includes) "$root/src/python_shim.c" \
    -L"$root/dist" -Wl,-rpath,'$ORIGIN/../../dist' -Wl,-l:libmojo-pyfastx.so \
    -o "$root/python/mojo_pyfastx/_native${suffix}"
