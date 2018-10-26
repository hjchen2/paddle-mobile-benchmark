#!/bin/sh

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/mobilenet_v2/mobilenet_v2.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/mobilenet_v2/mobilenet_v2_deploy.prototxt

python ./output.py ./output.npy ./fluid_mobilenet_v2 fc7


