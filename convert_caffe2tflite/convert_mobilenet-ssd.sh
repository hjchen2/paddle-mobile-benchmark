#!/bin/sh

# package tf model and graph file into single graph pb file
#../bazel-bin/tensorflow/python/tools/freeze_graph \
#  --input_graph=./mobilenet-v2/mobilenet_v2_1.0_224_eval.pbtxt \
#  --input_checkpoint=./mobilenet-v2/mobilenet_v2_1.0_224.ckpt \
#  --output_graph=mobilenet-v2/frozen_eval_graph.pb \
#  --output_node_names=MobilenetV2/Predictions/Reshape_1

# convert graph pb file to tflite format
#../bazel-bin/tensorflow/contrib/lite/toco/toco \
#  --input_file=./mobilenet-ssd/frozen_inference_graph.pb \
#  --output_file=./mobilenet-ssd/mobilenet-ssd.tflite \
#  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
#  --inference_type=FLOAT \
#  --input_shape="1,300,300,3" \
#  --input_array=input \
#  --output_array=MobilenetV2/Predictions/Reshape_1


# reference https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193
export CONFIG_FILE=./mobilenet-ssd/pipeline.config
export CHECKPOINT_PATH=./mobilenet-ssd/model.ckpt
export OUTPUT_DIR=./mobilenet-ssd/

python ../models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path=$CONFIG_FILE \
  --trained_checkpoint_prefix=$CHECKPOINT_PATH \
  --output_directory=$OUTPUT_DIR \
  --add_postprocessing_op=true

../bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=$OUTPUT_DIR/tflite_graph.pb \
  --output_file=$OUTPUT_DIR/mobilenet-ssd.tflite \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --inference_type=FLOAT \
  --change_concat_input_ranges=false \
  --allow_custom_ops
