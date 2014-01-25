import os

OpenCVInstallDir="ext/opencv_install/"
OpenCVPkgConfigDir = OpenCVInstallDir+"/lib/pkgconfig/"

env = Environment()
#env["CXX"] = "clang++"
env.Append(CXXFLAGS=['-std=c++11', '-O3', '-Wall', '-Werror'])
env.Append(CXXFLAGS=['-lto'])

env.ParseConfig("PKG_CONFIG_PATH="+OpenCVPkgConfigDir+" pkg-config opencv --cflags --libs --static")

env.Program('wipeout', 'main.cpp')
