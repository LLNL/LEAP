#!/bin/bash
# > ./etc/cpu_build.sh
 
# exit when any command fails
set -e

# Clean build directory
rm -rf ./cpu_build
mkdir -p cpu_build || echo 0
cd cpu_build

# Build LEAP
cmake -DCPUONLY=ON ..
cmake --build . -j8
