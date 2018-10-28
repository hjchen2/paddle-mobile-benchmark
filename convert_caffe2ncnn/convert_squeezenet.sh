rm -rf ../ncnn_models/squeezenet_v1.1
mkdir -p ../ncnn_models/squeezenet_v1.1

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/squeezenet_v1.1/deploy.prototxt \
  ../caffe_models/squeezenet_v1.1/squeezenet_v1.1.caffemodel \
  ../ncnn_models/squeezenet_v1.1/squeezenet_v1.1.param \
  ../ncnn_models/squeezenet_v1.1/squeezenet_v1.1.bin

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/squeezenet_v1.1/deploy.prototxt \
  ../caffe_models/squeezenet_v1.1/squeezenet_v1.1.caffemodel \
  ../ncnn_models/squeezenet_v1.1/squeezenet_v1.1_8bit.param \
  ../ncnn_models/squeezenet_v1.1/squeezenet_v1.1_8bit.bin \
  256 ../caffe_models/squeezenet_v1.1/squeezenet_v1.1_quant.table
