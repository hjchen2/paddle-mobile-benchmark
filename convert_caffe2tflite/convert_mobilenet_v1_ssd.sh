#!/bin/sh

# reference https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193
# and https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
export PYTHONPATH=../../tensorflow/models/research/:../../tensorflow/models/research/slim/:${PYTHONPATH}
export CONFIG_FILE=../tf_models/mobilenet_v1_ssd/pipeline.config
export CHECKPOINT_PATH=../tf_models/mobilenet_v1_ssd/model.ckpt
export OUTPUT_DIR=../tflite_models/mobilenet_v1_ssd/

rm -rf ${OUTPUT_DIR} && mkdir -p ${OUTPUT_DIR}

python ../../tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path=$CONFIG_FILE \
  --trained_checkpoint_prefix=$CHECKPOINT_PATH \
  --output_directory=$OUTPUT_DIR \
  --add_postprocessing_op=true

../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=$OUTPUT_DIR/tflite_graph.pb \
  --output_file=$OUTPUT_DIR/mobilenet_v1_ssd.tflite \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --inference_type=FLOAT \
  --change_concat_input_ranges=false \
  --allow_custom_ops

../../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=$OUTPUT_DIR/tflite_graph.pb \
  --output_file=$OUTPUT_DIR/mobilenet_v1_ssd_8bit.tflite \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --inference_type=QUANTIZED_UINT8 \
  --default_ranges_min=0.0 \
  --default_ranges_max=255.0 \
  --change_concat_input_ranges=false \
  --allow_custom_ops

rm -rf ${OUTPUT_DIR}/tflite_graph.pb ${OUTPUT_DIR}/tflite_graph.pbtxt
