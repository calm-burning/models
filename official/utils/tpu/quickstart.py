#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver


def cloud_tpu_resolution(tpu_name):
  """Demonstration of how to interact with gcloud metadata for TPUs.

  The model for cloud TPUs is fundamentally different from normal TensorFlow
  usage (with the arguable exception of distributed training). The user has
  total control of a cloud instance (sometimes referred to as the red VM), and
  uses TensorFlow to send instructions over a network to a Google managed
  instance which is connected to a TPU (sometimes referred to as the green VM).

  As a result in order to function code which is intended to run on TPUs must
  perform some network resolution. There are two supported methods:
    1)  REST calls
    2)  tensorflow.contrib.cluster_resolver.TPUClusterResolver

  The TPUClusterResolver simply wraps the REST calls, so there is no need to
  make requests directly.

  Args:
    tpu_name: A string with the name of the TPU
  """
  resolver = TPUClusterResolver(tpu=[tpu_name])
  tpu_url = resolver.get_master()
  print("TPU URL: {}".format(tpu_url))
  return tpu_url


def basic_operations(tpu_url):
  g = tf.Graph()
  with g.as_default():
    alpha = tf.Variable(3.0, name="alpha")
    beta = alpha + 1

  with tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    print(beta.eval())

  with tf.Session(tpu_url) as sess:
    tf.global_variables_initializer().run()


class DemoParser(argparse.ArgumentParser):
  def __init__(self):
    super(DemoParser, self).__init__()
    self.add_argument("--tpu", default="{}-tpu-0".format(os.getlogin()),
                      help="[default: %(default)s] The name of the TPU.",
                      metavar="<TPU>")

def main():
  parser = DemoParser()
  tpu_name = parser.parse_args(sys.argv[1:]).tpu
  tpu_url = cloud_tpu_resolution(tpu_name=tpu_name)
  basic_operations(tpu_url=tpu_url)

if __name__ == "__main__":
  main()
