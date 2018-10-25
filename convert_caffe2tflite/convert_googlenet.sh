#!/bin/sh

# convert caffe model to tensorflow model
python caffe-tensorflow/convert.py \
  --caffemodel=./googlenet/bvlc_googlenet.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  --standalone-output-path=./googlenet.pb \
  ./googlenet/deploy_remove_lrn.prototxt

# convert graph pb file to tflite format
# float
../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./googlenet.pb \
  --output_file=../tflite-models/googlenet/googlenet.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,224,224,3" \
  --input_array=data \
  --output_array=loss3_classifier/loss3_classifier

# uint8
../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./googlenet.pb \
  --output_file=../tflite-models/googlenet/googlenet_8bit.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224,224,3" \
  --input_array=data \
  --output_array=loss3_classifier/loss3_classifier \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0
