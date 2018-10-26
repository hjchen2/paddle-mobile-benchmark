#!/bin/sh

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/mobilenet_v1/mobilenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/mobilenet_v1/mobilenet_deploy.prototxt

python ./output.py ./output.npy ./fluid_mobilenet_v1 fc7


