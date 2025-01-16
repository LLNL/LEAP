rd /s /q win_build
mkdir win_build
cd win_build

cmake .. -G "Visual Studio 16" -DDEV_MODE=1
::cmake .. -G "Visual Studio 17" -DDEV_MODE=1
cmake --build . --config Release
