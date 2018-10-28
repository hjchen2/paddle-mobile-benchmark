#!/bin/sh

rm -rf ../fluid_models/mobilenet_v1
mkdir -p ../fluid_models/mobilenet_v1

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/mobilenet_v1/mobilenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/mobilenet_v1/mobilenet_deploy.prototxt

python ./output.py ./output.npy ../fluid_models/mobilenet_v1/mobilenet_v1 fc7
rm -rf output.*

# convert to int8 model
export PYTHONPATH=../:./:${PYTHONPATH}
python ./convert_to_quantize.py \
  --model=../fluid_models/mobilenet_v1/mobilenet_v1 \
  --output_ops="fc7.conv2d.output.1.tmp_1" \
  --output=../fluid_models/mobilenet_v1/mobilenet_v1_8bit
