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
from quantize_transpiler import _original_var_name
from quantize_transpiler import QuantizeTranspiler
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu', bool, False, "Whether to use GPU or not.")
add_arg('model', str, None, "Pretrained model path.", required=True)
add_arg('output', str, None, "Output directory for saved model.", required=True)
add_arg('input_ops', str, None, "Input ops.")
add_arg('output_ops', str, None, "Output ops.")
args = parser.parse_args()

def NextOps(start, program):
  """
  Get all next operators, return as operators list.

  Args:
      start(Operator): The start operator.
      program(Program): The program which has all operators.
  
  Return:
     List: An operators list.
  """
  next_ops = list()
  for output in start.output_names:
    output_var = start.output(output)
    for block in program.blocks:
      for op in block.ops:
        input_vars = [op.input(name) for name in op.input_names]
        if output_var in input_vars:
          next_ops.append(op)
  return next_ops

def RemoveFetchOp(program):
  """
  Remove fetch operators in origin program.

  Args: program(Program): The program which has all operators.
  
  Return:
      Program: The program which fetch operators have been removed.
  """
  for block in program.blocks:
    ops = list(block.ops)
    for op in ops:
      if op.type == "fetch":
        idx = ops.index(op)
        block._remove_op(idx)
  return program

def CorrectQuantOpName(program):
  """
  Correct fake quantize and dequantize op name. Operators with fake dequantize
  name will be renamed as dequantize, and with fake quantize name will be
  renamed as quantize.

  Args:
      program(Program): The program which has all operators.

  Return:
      Program: The corrected program.
  """
  for block in program.blocks:
    for op in list(block.ops):
      if op.type == "fake_dequantize_max_abs":
        op.desc.set_type("dequantize")
      if op.type == "fake_quantize_abs_max" or \
         op.type == "fake_quantize_range_abs_max":
        op.desc.set_type("quantize")
  return program

def FuseQuantAndConvPad(program):
  """
  Move convolution padding attr into quantize operator since padding in quantize
  is supported. It will be more efficient by preposing pad in paddle mobile.

  Args:
      program(Program): The program which has all operators.
  
  Return:
      Program: The Fused program.
  """
  for block in program.blocks:
    for op in list(block.ops):
      if op.type == "quantize":
        next_ops = NextOps(op, program)
        paddings = list()
        for next_op in next_ops:
          if next_op.has_attr("paddings"):
            paddings.append(next_op.attr("paddings"))
        if len(paddings) == 0:
          continue
        # fuse will be done only if paddings of the output operators are all the same
        if all(p == paddings[0] for p in paddings):
          op._set_attr("paddings", paddings[0])
          for next_op in next_ops:
            next_op._set_attr("paddings", [0, 0])
  return program

def convert():
  place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
  exe = fluid.Executor(place)
  [program, feed, fetch] = fluid.io.load_inference_model(args.model, exe)

  # set feed and fetch list
  if args.input_ops is not None:
    feed_list = args.input_ops.split(',')
    if len(feed_list) > 0:
      feed = [fluid.framework._get_var(var, program) for var in feed_list]
  if args.output_ops is not None:
    fetch_list = args.output_ops.split(',')
    if len(fetch_list) > 0:
      fetch = [fluid.framework._get_var(var, program) for var in fetch_list]
    if len(fetch) > 0:
      program = RemoveFetchOp(program)

  # quantize weights and save model
  quant_transpiler = QuantizeTranspiler()
  quant_transpiler.training_transpile(program)

  with fluid.program_guard(program):
    quant_transpiler.freeze_program(program, place)
    quant_transpiler.convert_to_int8(program, place)

  program = CorrectQuantOpName(program)
#  program = FuseQuantAndConvPad(program)
  fluid.io.save_inference_model(args.output, feed, fetch, exe, program)

if __name__ == '__main__':
    convert()
