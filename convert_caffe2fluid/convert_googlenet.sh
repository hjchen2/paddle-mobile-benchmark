#!/bin/sh

rm -rf ../fluid_models/googlenet_v1
mkdir -p ../fluid_models/googlenet_v1

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/googlenet/bvlc_googlenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/googlenet/deploy_remove_lrn.prototxt

python ./output.py ./output.npy ../fluid_models/googlenet_v1/googlenet_v1 loss3_classifier
rm -rf output.*

# convert to int8 model
export PYTHONPATH=../:./:${PYTHONPATH}
python ./convert_to_quantize.py \
  --model=../fluid_models/googlenet_v1/googlenet_v1 \
  --output_ops="loss3_classifier.fc.output.1.tmp_1" \
  --output=../fluid_models/googlenet_v1/googlenet_v1_8bit

