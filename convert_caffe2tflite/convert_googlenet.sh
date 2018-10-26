#!/bin/sh

# convert caffe model to tensorflow model
python caffe-tensorflow/convert.py \
  --caffemodel=../caffe_models/googlenet/bvlc_googlenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  --standalone-output-path=./googlenet.pb \
  ../caffe_models/googlenet/deploy_remove_lrn.prototxt


# convert graph pb file to tflite format
rm -rf ../tflite_models/googlenet_v1
mkdir -p ../tflite_models/googlenet_v1

# float
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./googlenet.pb \
  --output_file=../tflite_models/googlenet_v1/googlenet_v1.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,224,224,3" \
  --input_array=data \
  --output_array=prob

# uint8
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./googlenet.pb \
  --output_file=../tflite_models/googlenet_v1/googlenet_v1_8bit.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224,224,3" \
  --input_array=data \
  --output_array=prob \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0

# rm -rf googlenet.pb ./output.*
