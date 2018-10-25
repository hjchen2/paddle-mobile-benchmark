#!/bin/sh

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=./googlenet/bvlc_googlenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ./googlenet/deploy_remove_lrn.prototxt

python ./output.py ./output.npy ./fluid loss3_classifier


