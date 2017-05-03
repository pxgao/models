# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple GPU's with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = inception.inference(images, num_classes, for_training=True,
                                 restore_logits=restore_logits,
                                 scope=scope)

  return logits

def train(dataset):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
    num_classes = dataset.num_classes() + 1
    
    # Calculate the gradients for each model tower.
    tower_grads = []
    reuse_variables = None
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          # Force all Variables to reside on the CPU.
          with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
            # Calculate the loss for one tower of the ImageNet model. This
            # function constructs the entire ImageNet model but shares the
            # variables across all towers.
            image_shape = (FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3)
            labels_shape = (FLAGS.batch_size)
            images = tf.zeros(image_shape, dtype=tf.float32)
            labels = tf.zeros(labels_shape, dtype=tf.int32)
            
            loss = _tower_loss(images, labels, num_classes,
                               scope, reuse_variables)

          # Reuse variables for the next tower.
          reuse_variables = True

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      loss_value = sess.run(loss)
      duration = time.time() - start_time
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d, loss =(%.1f examples/sec; %.3f '
                      'sec/batch)')
      print(format_str % (datetime.now(), step,
                          examples_per_sec, duration))
