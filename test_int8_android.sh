#!/usr/bin/sh

if [ $# -ne 3 ]; then
  echo "Usage: ./test_int8_android.sh project_root_path model_path image_path"
  exit 1
fi

project_root_path=$1
model_path=$2
image_path=$3

adb shell rm -rf /data/local/tmp/*
adb shell mkdir /data/local/tmp/bin /data/local/tmp/models /data/local/tmp/images
adb push ${project_root_path}/build/release/arm-v7a/build/libpaddle-mobile.so /data/local/tmp/bin/
adb push ${project_root_path}/test/build/test-googlenet /data/local/tmp/bin/

adb push ${image_path} /data/local/tmp/images/test_image_1x3x224x224_float
adb push ${model_path} /data/local/tmp/models/googlenet

adb shell "cd /data/local/tmp/bin; LD_LIBRARY_PATH=. ./test-googlenet"
