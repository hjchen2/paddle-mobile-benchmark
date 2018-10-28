#!/bin/sh

rm -rf ../fluid_models/mobilenet_v2
mkdir -p ../fluid_models/mobilenet_v2

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/mobilenet_v2/mobilenet_v2.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/mobilenet_v2/mobilenet_v2_deploy.prototxt

python ./output.py ./output.npy ../fluid_models/mobilenet_v2/mobilenet_v2 fc7
rm -rf output.*

# convert to int8 model
export PYTHONPATH=../:${PYTHONPATH}
python ./convert_to_quantize.py \
  --model=../fluid_models/mobilenet_v2/mobilenet_v2 \
  --output_ops="fc7.conv2d.output.1.tmp_1" \
  --output=../fluid_models/mobilenet_v2/mobilenet_v2_8bit
