rd /s /q win_build
mkdir win_build
cd win_build

cmake .. -DDEV_MODE=1
cmake --build . --config Release
