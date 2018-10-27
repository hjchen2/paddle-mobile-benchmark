#!/usr/bin/sh

rm -rf ../ncnn_models/mobilenet_v2
mkdir -p ../ncnn_models/mobilenet_v2

# float
../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v2/mobilenet_v2_deploy.prototxt \
  ../caffe_models/mobilenet_v2/mobilenet_v2.caffemodel \
  ../ncnn_models/mobilenet_v2/mobilenet_v2.param \
  ../ncnn_models/mobilenet_v2/mobilenet_v2.bin

# 8bit quant
../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v2/mobilenet_v2_deploy.prototxt \
  ../caffe_models/mobilenet_v2/mobilenet_v2.caffemodel \
  ../ncnn_models/mobilenet_v2/mobilenet_v2_8bit.param \
  ../ncnn_models/mobilenet_v2/mobilenet_v2_8bit.bin \
  256 ../caffe_models/mobilenet_v2/mobilenet_v2_quant.table
