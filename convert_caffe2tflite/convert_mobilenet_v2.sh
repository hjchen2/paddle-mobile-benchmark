#!/bin/sh

# package tf model and graph file into single graph pb file
# if you use the model from tensorflow models, you should uncomment this.
#../bazel-bin/tensorflow/python/tools/freeze_graph \
#  --input_graph=./mobilenet-v2/mobilenet_v2_1.0_224_eval.pbtxt \
#  --input_checkpoint=./mobilenet-v2/mobilenet_v2_1.0_224.ckpt \
#  --output_graph=./frozen_eval_graph.pb \
#  --output_node_names=MobilenetV2/Predictions/Reshape_1

# convert caffe model to tensorflow model
python caffe-tensorflow/convert.py \
  --caffemodel=../caffe_models/mobilenet_v2/mobilenet_v2.caffemodel \
  --data-output-path=./output.npy \
  --code-output-path=./output.py \
  --standalone-output-path=./frozen_eval_graph.pb \
  ../caffe_models/mobilenet_v2/mobilenet_v2_deploy.prototxt

# convert graph pb file to tflite format
rm -rf ../tflite_models/mobilenet-v2
mkdir -p ../tflite_models/mobilenet-v2

# float
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./frozen_eval_graph.pb \
  --output_file=../tflite_models/mobilenet-v2/mobilenet_v2.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,224,224,3" \
  --input_array=input \
  --output_array=fc7
#  --output_array=MobilenetV2/Predictions/Reshape_1

# uint8
../../tensorflow-lite/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./frozen_eval_graph.pb \
  --output_file=../tflite_models/mobilenet-v2/mobilenet_v2_8bit.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224,224,3" \
  --input_array=data \
  --output_array=fc7 \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0
#  --output_array=MobilenetV2/Predictions/Reshape_1
