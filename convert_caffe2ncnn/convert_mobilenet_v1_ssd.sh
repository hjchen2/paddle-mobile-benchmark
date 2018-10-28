#!/usr/bin/sh

rm -rf ../ncnn_models/mobilenet_v1_ssd
mkdir -p ../ncnn_models/mobilenet_v1_ssd

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v1_ssd/deploy.prototxt \
  ../caffe_models/mobilenet_v1_ssd/mobilenet_iter_73000.caffemodel \
  ../ncnn_models/mobilenet_v1_ssd/mobilenet_v1_ssd.param \
  ../ncnn_models/mobilenet_v1_ssd/mobilenet_v1_ssd.bin

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/mobilenet_v1_ssd/deploy.prototxt \
  ../caffe_models/mobilenet_v1_ssd/mobilenet_iter_73000.caffemodel \
  ../ncnn_models/mobilenet_v1_ssd/mobilenet_v1_ssd_8bit.param \
  ../ncnn_models/mobilenet_v1_ssd/mobilenet_v1_ssd_8bit.bin \
  256 ../caffe_models/mobilenet_v1_ssd/mobilenet_ssd_quant.table
