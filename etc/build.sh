#!/bin/bash
# > ./etc/build.sh
 
# exit when any command fails
set -e

# Clean build directory
rm -rf ./build
mkdir -p build || echo 0
cd build

# Build LEAP
cmake ..
cmake --build . -j8
