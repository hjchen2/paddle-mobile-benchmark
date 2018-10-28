#!/bin/sh

# package tf model and graph file into single graph pb file
# if you use the model from tensorflow models, you should uncomment this.
#../../tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
#  --input_graph=../tf_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_eval.pbtxt \
#  --input_checkpoint=../tf_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
#  --output_graph=./frozen_eval_graph.pb \
#  --output_node_names=MobilenetV1/Predictions/Reshape_1

# convert caffe model to tensorflow model
#python caffe-tensorflow/convert.py \
#  --caffemodel=../caffe_models/mobilenet_v1/mobilenet.caffemodel \
#  --data-output-path=./output.npy \
#  --code-output-path=./output.py \
#  --standalone-output-path=./frozen_eval_graph.pb \
#  ../caffe_models/mobilenet_v1/mobilenet_deploy.prototxt

cp ../tf_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb ./frozen_eval_graph.pb

# convert graph pb file to tflite format
rm -rf ../tflite_models/mobilenet_v1
mkdir -p ../tflite_models/mobilenet_v1

# float
../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./frozen_eval_graph.pb \
  --output_file=../tflite_models/mobilenet_v1/mobilenet_v1.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,224,224,3" \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
#  --output_array=MobilenetV1/Predictions/Reshape_1

# uint8
../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=./frozen_eval_graph.pb \
  --output_file=../tflite_models/mobilenet_v1/mobilenet_v1_8bit.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224,224,3" \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0
#  --output_array=MobilenetV1/Predictions/Reshape_1
