#!/bin/bash

set -e -u

# run this from ext/opencv-src/build, then make -j4 then make install
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=../../opencv_install
