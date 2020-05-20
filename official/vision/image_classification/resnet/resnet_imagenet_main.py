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

import math
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.modeling import performance
from official.staging.training import controller
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_runnable

flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                          'tf.function.')
flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                     help='Calculate L2_loss on concatenated weights, '
                          'instead of using Keras per-layer L2 loss.')


def build_stats(runnable, time_callback):
    """Normalizes and returns dictionary of stats.

    Args:
      runnable: The module containing all the training and evaluation metrics.
      time_callback: Time tracking callback instance.

    Returns:
      Dictionary of normalized results.
    """
    stats = {}

    if not runnable.flags_obj.skip_eval:
        stats['eval_loss'] = runnable.test_loss.result().numpy()
        stats['eval_acc'] = runnable.test_accuracy.result().numpy()

        stats['train_loss'] = runnable.train_loss.result().numpy()
        stats['train_acc'] = runnable.train_accuracy.result().numpy()

    if time_callback:
        timestamp_log = time_callback.timestamp_log
        stats['step_timestamp_log'] = timestamp_log
        stats['train_finish_time'] = time_callback.train_finish_time
        if time_callback.epoch_runtime_log:
            stats['avg_exp_per_second'] = time_callback.average_examples_per_second

    return stats


def get_num_train_iterations(flags_obj):
    """Returns the number of training steps, train and test epochs."""
    train_steps = (
            imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
    train_epochs = flags_obj.train_epochs

    if flags_obj.train_steps:
        train_steps = min(flags_obj.train_steps, train_steps)
        train_epochs = 1

    eval_steps = math.ceil(1.0 * imagenet_preprocessing.NUM_IMAGES['validation'] /
                           flags_obj.batch_size)

    return train_steps, train_epochs, eval_steps

def get_num_train_iterations_non_flag(batch_size, train_epochs, num_images_train, num_images_val, provided_train_steps=None):

    """Returns the number of training steps, train and test epochs."""
    train_steps = (
            num_images_train // batch_size)
    train_epochs = train_epochs

    if provided_train_steps:
        train_steps = min(provided_train_steps, train_steps)
        train_epochs = 1

    eval_steps = math.ceil(1.0 * num_images_val / batch_size)

    return train_steps, train_epochs, eval_steps


def _steps_to_run(steps_in_current_epoch, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    return min(steps_per_loop, steps_per_epoch - steps_in_current_epoch)


def run():
    """Run ResNet ImageNet training and eval loop using custom training loops.

    Args:
      flags_obj: An object containing parsed flag values.

    Raises:
      ValueError: If fp16 is passed as it is not currently supported.

    Returns:
      Dictionary of training and eval stats.
    """
    enable_xla = False
    dtype= tf.float16
    tf_gpu_thread_mode = None
    per_gpu_thread_count = 0
    tf_gpu_thread_mode = None
    num_gpus = 1
    datasets_num_private_threads = None
    data_format = None
    distribution_strategy = 'mirrored'
    num_gpus = 1
    all_reduce_alg = None
    num_packs = 1
    tpu = None
    batch_size = 32
    train_epochs = 1
    steps_per_loop = 500
    log_steps = 100
    model_dir = '/tmp'
    enable_tensorboard = False
    use_tf_while_loop = True
    use_tf_function = True
    use_synthetic_data = False
    single_l2_loss_op = False
    fp16_implementation = 'keras'
    use_tf_function = True
    loss_scale = None
    single_l2_loss_op = False
    epochs_between_evals = 1
    enable_checkpoint_and_export = False
    enable_tensorboard = False
    skip_eval = False


    keras_utils.set_session_config(
        enable_xla=enable_xla)
    performance.set_mixed_precision_policy(dtype)

    if tf.config.list_physical_devices('GPU'):
        if tf_gpu_thread_mode:
            keras_utils.set_gpu_thread_mode_and_count(
                per_gpu_thread_count=per_gpu_thread_count,
                gpu_thread_mode=tf_gpu_thread_mode,
                num_gpus=num_gpus,
                datasets_num_private_threads=datasets_num_private_threads)
        common.set_cudnn_batchnorm_mode()

    # TODO(anj-s): Set data_format without using Keras.
    if data_format is None:
        data_format = ('channels_first' if tf.config.list_physical_devices('GPU')
                       else 'channels_last')
    tf.keras.backend.set_image_data_format(data_format)

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=distribution_strategy,
        num_gpus=num_gpus,
        all_reduce_alg=all_reduce_alg,
        num_packs=num_packs,
        tpu_address=None)

    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations(
        flags_obj)
    steps_per_loop = min(steps_per_loop, per_epoch_steps)

    logging.info(
        'Training %d epochs, each epoch has %d steps, '
        'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
        train_epochs * per_epoch_steps, eval_steps)

    time_callback = keras_utils.TimeHistory(
        batch_size,
        log_steps,
        logdir=model_dir if enable_tensorboard else None)
    with distribution_utils.get_strategy_scope(strategy):
        runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback,
                                                  per_epoch_steps)

    eval_interval = epochs_between_evals * per_epoch_steps
    checkpoint_interval = (
        per_epoch_steps if enable_checkpoint_and_export else None)
    summary_interval = per_epoch_steps if enable_tensorboard else None

    checkpoint_manager = tf.train.CheckpointManager(
        runnable.checkpoint,
        directory=model_dir,
        max_to_keep=10,
        step_counter=runnable.global_step,
        checkpoint_interval=checkpoint_interval)

    resnet_controller = controller.Controller(
        strategy,
        runnable.train,
        runnable.evaluate if not skip_eval else None,
        global_step=runnable.global_step,
        steps_per_loop=steps_per_loop,
        train_steps=per_epoch_steps * train_epochs,
        checkpoint_manager=checkpoint_manager,
        summary_interval=summary_interval,
        eval_steps=eval_steps,
        eval_interval=eval_interval)

    time_callback.on_train_begin()
    resnet_controller.train(evaluate=not skip_eval)
    time_callback.on_train_end()

    stats = build_stats(runnable, time_callback)
    return stats

def debugflags(flags_obj):

    print("enable_xla=", flags_obj.enable_xla)
    print("dtype=", flags_obj.dtype) #performance.set_mixed_precision_policy(flags_core.get_tf_dtype(flags_obj))
    print("tf_gpu_thread_mode=", flags_obj.tf_gpu_thread_mode)
    print("per_gpu_thread_count=", flags_obj.per_gpu_thread_count)
    print("tf_gpu_thread_mode=", flags_obj.tf_gpu_thread_mode)
    print("num_gpus=", flags_obj.num_gpus)
    print("datasets_num_private_threads=", flags_obj.datasets_num_private_threads)
    print("data_format=", flags_obj.data_format)
    print("distribution_strategy=", flags_obj.distribution_strategy)
    print("num_gpus=", flags_obj.num_gpus)
    print("all_reduce_alg=", flags_obj.all_reduce_alg)
    print("num_packs=", flags_obj.num_packs)
    print("tpu=", flags_obj.tpu)
    print("batch_size=", flags_obj.batch_size)
    print("train_epochs=", flags_obj.train_epochs)
    #per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations_non_flag(batch_size, train_epochs, num_images_train, num_images_val, provided_train_steps=None)
    print("steps_per_loop=", flags_obj.steps_per_loop)
    print("log_steps=", flags_obj.log_steps)
    print("model_dir=", flags_obj.model_dir)
    print("enable_tensorboard=", flags_obj.enable_tensorboard)
    #########################
    #runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback, per_epoch_steps)
    print("use_tf_while_loop=", flags_obj.use_tf_while_loop)
    print("use_tf_function=", flags_obj.use_tf_function)
    print("dtype=", flags_obj.dtype) #dtype = flags_core.get_tf_dtype(flags_obj)
    #print("batch_size=", flags_obj.batch_size)
    print("use_synthetic_data=", flags_obj.use_synthetic_data)
    print("single_l2_loss_op=", flags_obj.single_l2_loss_op)
    print("fp16_implementation=",flags_obj.fp16_implementation)
    print("use_tf_function=", flags_obj.use_tf_function)
    print("loss_scale=", flags_obj.loss_scale)
    print("data_dir=", flags_obj.data_dir)
    print("single_l2_loss_op=", flags_obj.single_l2_loss_op)
    #######
    print("epochs_between_evals=", flags_obj.epochs_between_evals)
    print("enable_checkpoint_and_export=", flags_obj.enable_checkpoint_and_export)
    print("enable_tensorboard=", flags_obj.enable_tensorboard)
    print("skip_eval=", flags_obj.skip_eval)
    return None


def main(_):
    model_helpers.apply_clean(flags.FLAGS)
    stats = debugflags(flags.FLAGS) #run()
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    common.define_keras_flags()
    app.run(main)
