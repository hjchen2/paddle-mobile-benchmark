#!/bin/sh

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/googlenet/bvlc_googlenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/googlenet/deploy_remove_lrn.prototxt

python ./output.py ./output.npy ./fluid_googlenet_v1 loss3_classifier


