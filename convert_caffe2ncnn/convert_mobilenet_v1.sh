#!/usr/bin/sh

rm -rf ../ncnn_models/mobilenet_v1
mkdir -p ../ncnn_models/mobilenet_v1

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v1/mobilenet_deploy.prototxt \
  ../caffe_models/mobilenet_v1/mobilenet.caffemodel \
  ../ncnn_models/mobilenet_v1/mobilenet_v1.param \
  ../ncnn_models/mobilenet_v1/mobilenet_v1.bin

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v1/mobilenet_deploy.prototxt \
  ../caffe_models/mobilenet_v1/mobilenet.caffemodel \
  ../ncnn_models/mobilenet_v1/mobilenet_v1_8bit.param \
  ../ncnn_models/mobilenet_v1/mobilenet_v1_8bit.bin \
  256 ../caffe_models/mobilenet_v1/mobilenet_quant.table
