#!/bin/sh

# convert caffe model to tensorflow model
python caffe-tensorflow/convert.py \
  --caffemodel=../caffe_models/squeezenet_v1.1/squeezenet_v1.1.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  --standalone-output-path=./squeezenet_v1.1.pb \
  ../caffe_models/squeezenet_v1.1/deploy.prototxt


# convert graph pb file to tflite format
rm -rf ../tflite_models/squeezenet_v1.1
mkdir -p ../tflite_models/squeezenet_v1.1

# float
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./squeezenet_v1.1.pb \
  --output_file=../tflite_models/squeezenet_v1.1/squeezenet_v1.1.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,227,227,3" \
  --input_array=data \
  --output_array=prob

# uint8
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./squeezenet_v1.1.pb \
  --output_file=../tflite_models/squeezenet_v1.1/squeezenet_v1.1_8bit.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,227,227,3" \
  --input_array=data \
  --output_array=prob \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0

rm -rf squeezenet_v1.1.pb ./output.*
