#!/bin/sh

# package tf model and graph file into single graph pb file
../bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=./mobilenet-v2/mobilenet_v2_1.0_224_eval.pbtxt \
  --input_checkpoint=./mobilenet-v2/mobilenet_v2_1.0_224.ckpt \
  --output_graph=mobilenet-v2/frozen_eval_graph.pb \
  --output_node_names=MobilenetV2/Predictions/Reshape_1

# convert graph pb file to tflite format
../bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./mobilenet-v2/frozen_eval_graph.pb \
  --output_file=./mobilenet-v2/mobilenet-v2.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,224,224,3" \
  --input_array=input \
  --output_array=MobilenetV2/Predictions/Reshape_1
