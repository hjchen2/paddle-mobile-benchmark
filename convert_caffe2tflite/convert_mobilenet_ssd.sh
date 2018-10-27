#!/bin/sh

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
