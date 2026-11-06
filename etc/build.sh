#!/bin/bash
# > ./etc/build.sh
 
# exit when any command fails
set -e

# Clean build directory
rm -rf ./build
mkdir -p build || echo 0
cd build

# Build LEAP
# Honor an optional LEAP_USE_NOTEX override (ON/OFF); otherwise CMake auto-detects
# ROCm/HIP and selects the software texture backend accordingly.
CMAKE_ARGS=""
if [ -n "${LEAP_USE_NOTEX}" ]; then
  CMAKE_ARGS="-DLEAP_USE_NOTEX=${LEAP_USE_NOTEX}"
fi
cmake ${CMAKE_ARGS} ..
cmake --build . -j8
