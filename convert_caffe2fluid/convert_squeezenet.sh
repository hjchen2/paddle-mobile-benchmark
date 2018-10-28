#!/bin/sh

rm -rf ../fluid_models/squeezenet_v1.1
mkdir -p ../fluid_models/squeezenet_v1.1

# convert caffe model to fluid model
python caffe-fluid/convert.py \
  --caffemodel=../caffe_models/squeezenet_v1.1/squeezenet_v1.1.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  ../caffe_models/squeezenet_v1.1/deploy.prototxt

python ./output.py ./output.npy ../fluid_models/squeezenet_v1.1/squeezenet_v1.1 prob
rm -rf output.*

# convert to int8 model
export PYTHONPATH=../:./:${PYTHONPATH}
python ./convert_to_quantize.py \
  --model=../fluid_models/squeezenet_v1.1/squeezenet_v1.1 \
  --output_ops="prob.softmax.output.1.tmp_0" \
  --output=../fluid_models/squeezenet_v1.1/squeezenet_v1.1_8bit

