# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Model that gives us explicit control over the model size
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes

#Returns a list of logists and aux_logits tensors
def param_model(inputs,
                dropout_keep_prop=0.8,
                num_classes=1000,
                is_training=True,
                restore_logits=True,
                scope='',
                layers=1,
                random_shape=[2000, 2000, 2000]):

  end_points = {}
  with tf.op_scope([inputs], scope, 'param_model'):
    #Create layers number of layers
    for layer in range(layers):
      # mixed: 35 x 35 x 256.
      with tf.variable_scope('layer_{}'.format(layer)):
        with tf.variable_scope('branch_0'):
          end_points['random_{}'.format(layer)] = tf.zeros(random_shape, tf.float32)
    logits_size = [num_classes, 1]
    logits = tf.zeros(logits_size, tf.float32)
    endpoints['logits'] = logits

  return logits, end_points

#Is this ever called? possible unnecessary
def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
