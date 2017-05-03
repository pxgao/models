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

def train(dataset):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images, labels = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

     # Split the batch of images and labels for towers.
    images_splits = tf.split(0, FLAGS.num_gpus, images)
    labels_splits = tf.split(0, FLAGS.num_gpus, labels)

    # Calculate the gradients for each model tower.
    tower_grads = []
    reuse_variables = None
    all_assign_op = []
    assign_op = None
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          # Force all Variables to reside on the CPU.
          with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
            # Calculate the loss for one tower of the ImageNet model. This
            # function constructs the entire ImageNet model but shares the
            # variables across all towers.
            assign_op  = tf.identity(images_splits[i])
            all_assign_op.append(assign_op)
      
    with tf.control_dependencies(all_assign_op):
      dummy_op = tf.constant(5)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    #Sleep for 30 seconds to ensure that there is sufficient data loaded
    time.sleep(30)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      sess.run(dummy_op)
      #sess.run(assign_op)
      #for op in all_assign_op:
      #  result = sess.run(op)
      duration = time.time() - start_time

      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step,
                            examples_per_sec, duration))

