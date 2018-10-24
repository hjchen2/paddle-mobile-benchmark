#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import numpy as np
import six
import functools
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.quantize.quantize_transpiler import _original_var_name
from paddle.fluid.contrib.quantize.quantize_transpiler import QuantizeTranspiler
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('model', str, None, "Pretrained model path.")
add_arg('input_image', str, None, "Image to Predict")
add_arg('input_ops', str, None, "Input ops.")
add_arg('output_ops', str, None, "Output ops.")
args = parser.parse_args()


def infer():
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [program, feed, fetch] = fluid.io.load_inference_model(args.model, exe)
    # remove fetch ops in origin program
    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == "fetch":
                idx = ops.index(op)
                block._remove_op(idx)
    # set feed and fetch list
    if args.input_ops is not None:
      feed_list = args.input_ops.split(',')
      if len(feed_list) > 0:
        feed = [fluid.framework._get_var(var, program) for var in feed_list]
    if args.output_ops is not None:
      fetch_list = args.output_ops.split(',')
      if len(fetch_list) > 0:
        fetch = [fluid.framework._get_var(var, program) for var in fetch_list]
    # quantize weights
    quant_transpiler = QuantizeTranspiler()
    quant_transpiler.training_transpile(program)
    # read test image
    test_data = np.fromfile(args.input_image, dtype=np.float32)
    test_data = [[test_data.reshape([3, 224, 224])]]
    # infer
    with fluid.program_guard(program):
        quant_transpiler.freeze_program(program, place)
        feeder = fluid.DataFeeder(feed_list=feed, place=place)
        fetch_out = exe.run(program=program,
                            feed=feeder.feed(test_data),
                            fetch_list=fetch)
        # print result
        for out in fetch_out:
          stride = int((out.size + 19) / 20)
          loop = int(out.size / stride)
          for i in range(loop):
            print out.flat[i * stride],


if __name__ == '__main__':
    infer()
