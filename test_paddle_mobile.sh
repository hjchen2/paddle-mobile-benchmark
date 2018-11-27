#!/usr/bin/sh

Usage="
  Usage: ./test_paddle_mobile.sh project_root_path model_path image_path [thread_num] [optimize]\n
  params: \n
    project_root_path: paddle-mobile root path, for example ../paddle-mobile.\n \
    model_path: pretrained model used in prediction, for example ./fluid_models/googlenet_v1.\n \
    image_path: image file path, for example ./test_image.png.\n \
    thread_num: optional int, threads count while predicting, default is 1.\n \
    optimize: optional bool, use fusion optimization or not, default is 0.\n"

if [ $# -lt 3 ]; then
  echo $Usage
  exit 1
fi

project_root_path=$1
model_path=$2
image_path=$3
thread_num=1
optimize=0
if [ $# -eq 4 ]; then
  thread_num=$4
fi
if [ $# -eq 5 ]; then
  optimize=$5
fi

adb shell rm -rf /data/local/tmp/*
adb shell mkdir /data/local/tmp/bin /data/local/tmp/models /data/local/tmp/images
adb push ${project_root_path}/build/release/arm-v7a/build/libpaddle-mobile.so /data/local/tmp/bin/
adb push ${project_root_path}/test/build/test-googlenet /data/local/tmp/bin/

adb push ${image_path} /data/local/tmp/images/test_image_1x3x224x224_float
adb push ${model_path} /data/local/tmp/models/googlenet

adb shell "cd /data/local/tmp/bin; LD_LIBRARY_PATH=. ./test-googlenet $thread_num $optimize"
