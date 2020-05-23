# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.modeling import performance
from official.staging.training import grad_utils
from official.staging.training import standard_runnable
from official.staging.training import utils
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model

class ResnetHopsRunnable(standard_runnable.StandardTrainable,
                     standard_runnable.StandardEvaluable):
  """Implements the training and evaluation APIs for Resnet model."""



  def __init__(self, eval_input_fn, train_input_fn, use_tf_while_loop, use_tf_function, dtype, global_batch_size,
               datasets_num_private_threads, single_l2_loss_op, loss_scale, fp16_implementation, bytes_per_pack,
               num_images_train, time_callback, epoch_steps):
    standard_runnable.StandardTrainable.__init__(self,
                                                 use_tf_while_loop,
                                                 use_tf_function)
    standard_runnable.StandardEvaluable.__init__(self,
                                                 use_tf_function)

    self.strategy = tf.distribute.get_strategy()
    #self.flags_obj = flags_obj
    self.dtype = dtype
    self.time_callback = time_callback

    self.single_l2_loss_op = single_l2_loss_op
    self.loss_scale = loss_scale
    self.bytes_per_pack = bytes_per_pack

    # Input pipeline related
    self.global_batch_size = global_batch_size
    self.num_images_train = num_images_train
    self.datasets_num_private_threads = datasets_num_private_threads

    if global_batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(self.global_batch_size / self.strategy.num_replicas_in_sync)

    self.eval_input_fn = eval_input_fn
    self.train_input_fn = train_input_fn

    self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        use_l2_regularizer=not single_l2_loss_op)

    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=self.global_batch_size,
        epoch_size=self.num_images_train,
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)
    self.optimizer = common.get_optimizer(lr_schedule)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations

    use_graph_rewrite = fp16_implementation == 'graph_rewrite'
    if use_graph_rewrite and not use_tf_function:
      raise ValueError('--fp16_implementation=graph_rewrite requires '
                       '--use_tf_function to be true')
    self.optimizer = performance.configure_optimizer(
        self.optimizer,
        use_float16=self.dtype == tf.float16,
        use_graph_rewrite=use_graph_rewrite,
        loss_scale= _get_loss_scale(dtype, loss_scale,  default_for_fp16=128)) #flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    # Handling epochs.
    self.epoch_steps = epoch_steps
    self.epoch_helper = utils.EpochHelper(epoch_steps, self.global_step)

  def build_train_dataset(self):
      # TODO (davit): make this as util function
      return self.train_input_fn
  #   """See base class."""
  #   return utils.make_distributed_dataset(
  #       self.strategy,
  #       self.input_fn,
  #       is_training=True,
  #       data_dir=self.data_dir,
  #       batch_size=self.batch_size,
  #       parse_record_fn=imagenet_preprocessing.parse_record,
  #       datasets_num_private_threads=self.datasets_num_private_threads,
  #       dtype=self.dtype,
  #       drop_remainder=True)
  #
  def build_eval_dataset(self):
    """See base class."""
    # TODO (davit): make this as util function
    return self.eval_input_fn
    # return utils.make_distributed_dataset(
    #     self.strategy,
    #     self.input_fn,
    #     is_training=False,
    #     data_dir=self.data_dir,
    #     batch_size=self.batch_size,
    #     parse_record_fn=imagenet_preprocessing.parse_record,
    #     dtype=self.dtype)

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    self.train_loss.reset_states()
    self.train_accuracy.reset_states()

    self._epoch_begin()
    self.time_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.global_batch_size)
        num_replicas = self.strategy.num_replicas_in_sync
        l2_weight_decay = 1e-4
        if self.single_l2_loss_op:
          l2_loss = l2_weight_decay * 2 * tf.add_n([
              tf.nn.l2_loss(v)
              for v in self.model.trainable_variables
              if 'bn' not in v.name
          ])

          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(self.model.losses) / num_replicas)

      grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables,
          pre_allreduce_callbacks=None,
          post_allreduce_callbacks=None,
          bytes_per_pack=self.bytes_per_pack)
      self.train_loss.update_state(loss)
      self.train_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {
        'train_loss': self.train_loss.result(),
        'train_accuracy': self.train_accuracy.result(),
    }
    self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    self._epoch_end()
    return metrics

  def eval_begin(self):
    """See base class."""
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits = self.model(images, training=False)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
      loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    return {
        'test_loss': self.test_loss.result(),
        'test_accuracy': self.test_accuracy.result()
    }

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)

def _get_loss_scale(dtype, loss_scale,  default_for_fp16):
    if loss_scale == "dynamic":
        return loss_scale
    elif loss_scale is not None:
        return float(loss_scale)
    elif dtype == tf.float32 or dtype == tf.bfloat16:
        return 1  # No loss scaling is needed for fp32
    else:
        assert dtype == tf.float16
        return default_for_fp16