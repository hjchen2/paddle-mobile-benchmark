# tensorflow project root path
tf_root_path=../tensorflow-lite
# paddle mobile project root path
pm_root_path=../paddle-mobile
# ncnn project root path
ncnn_root_path=../ncnn

# tf model path
tf_model_path=./tflite-models/googlenet/googlenet.tflite
tf_8bit_model_path=./tflite-models/googlenet/googlenet_8bit.tflite

#paddle mobile model path
pm_model_path=./fluid-models/googlenet
pm_8bit_model_path=./fluid-models/googlenet_8bit

# image path
image_path=./test_image_1x3x224x224_float

# num threads
num_threads=1

if [ $# -ne  4 ]; then
  echo "Usage: ./test_benchmark.sh pm/tflite/ncnn model_path feed_shape num_thread"
  exit 1
fi

model_path=$2
feed_shape=$3
num_threads=$4

if [[ $1 == 'tflite' ]]; then
  adb push ${tf_root_path}/bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model /data/local/tmp
  adb shell chmod +x /data/local/tmp/benchmark_model
  adb push ${model_path} /data/local/tmp

  adb shell taskset f0 /data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/${model_path##*/} \
    --num_threads=${num_threads} \
    --use_nnapi=true
elif [[ $1 == 'pm' ]]; then
  adb shell rm -rf /data/local/tmp/*
  adb push ${pm_root_path}/build/release/arm-v7a/build/libpaddle-mobile.so /data/local/tmp/
  adb push ${pm_root_path}/test/build/test-benchmark /data/local/tmp/
  adb shell chmod +x /data/local/tmp/test-benchmark
  adb push ${model_path} /data/local/tmp/${model_path##*/}
  adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=. taskset f0 ./test-benchmark ./${model_path##*/} ${feed_shape} ${num_threads}"
elif [[ $1 == 'ncnn' ]]; then
  adb shell rm -rf /data/local/tmp/*
  adb push ${ncnn_root_path}/build-android-armv7/benchmark/benchncnn /data/local/tmp/
  adb shell chmod +x /data/local/tmp/benchncnn
  adb push ${model_path} /data/local/tmp/${model_path##*/}
  adb shell taskset f0 /data/local/tmp/benchncnn \
      /data/local/tmp/${model_path##*/} ${feed_shape} ${num_threads}
else
  echo "The first param should be one of pm(paddle-mobile), tflite and ncnn."
fi
