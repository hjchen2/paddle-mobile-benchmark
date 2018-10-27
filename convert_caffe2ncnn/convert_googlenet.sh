../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/googlenet/deploy_remove_lrn_upgrade.prototxt \
  ../caffe_models/googlenet/bvlc_googlenet_upgrade.caffemodel \
  ../ncnn_models/googlenet/googlenet.param \
  ../ncnn_models/googlenet/googlenet.bin

../../ncnn/build/tools/caffe/caffe2ncnn \
  ../caffe_models/googlenet/deploy_remove_lrn_upgrade.prototxt \
  ../caffe_models/googlenet/bvlc_googlenet_upgrade.caffemodel \
  ../ncnn_models/googlenet/googlenet_8bit.param \
  ../ncnn_models/googlenet/googlenet_8bit.bin \
  256 ../caffe_models/googlenet/googlenet_quant.table
