from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import logging

from official.modeling import performance
from official.staging.training import controller
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.image_classification.resnet import resnet_hops_runnable

def run(
        num_gpus = 16,

        global_batch_size = 512,
        train_epochs = 50,
        epochs_between_evals = 1,
        steps_per_loop = 200,
        log_steps = 100,

        enable_xla = False,
        single_l2_loss_op = True,
        dtype= tf.float16,
        fp16_implementation = 'keras',
        loss_scale =  'dynamic',
        num_packs = 8, # this is releveant only for 'mirrored'
        bytes_per_pack = 16 * 1024 * 1024,

        tf_gpu_thread_mode = 'gpu_private', #None
        batchnorm_spatial_persistent = True,
        distribution_strategy = 'multi_worker_mirrored', #'mirrored'
        all_reduce_alg = 'nccl', #None

        NCCL_SOCKET_NTHREADS = '8'

):

    """Run ResNet ImageNet training and eval loop using custom training loops.

    Args: ...

    Returns:
      Dictionary of training and eval stats.
    """
    import os
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_NTHREADS'] = NCCL_SOCKET_NTHREADS

    import tensorflow as tf

    import numpy as np

    from hops import tensorboard
    from hops import hdfs
    import pydoop.hdfs as pydoop

    # Utility module for getting number of GPUs accessible by the container (Spark Executor)
    from hops import devices

    import multiprocessing
    model_dir = tensorboard.logdir()


    def build_stats(runnable, skip_eval, time_callback):
        """Normalizes and returns dictionary of stats.

        Args:
          runnable: The module containing all the training and evaluation metrics.
          time_callback: Time tracking callback instance.

        Returns:
          Dictionary of normalized results.
        """
        stats = {}

        if not skip_eval:
            stats['eval_loss'] = runnable.test_loss.result().numpy()
            stats['eval_acc'] = runnable.test_accuracy.result().numpy()

            stats['train_loss'] = runnable.train_loss.result().numpy()
            stats['train_acc'] = runnable.train_accuracy.result().numpy()

        if time_callback:
            #timestamp_log = time_callback.timestamp_log
            #stats['step_timestamp_log'] = timestamp_log
            stats['train_finish_time'] = time_callback.train_finish_time
            stats['train_time'] = time_callback.elapsed_time
            if time_callback.epoch_runtime_log:
                stats['avg_exp_per_second'] = time_callback.average_examples_per_second

        return stats

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

    def get_datasets_num_private_threads(datasets_num_private_threads, num_gpus, per_gpu_thread_count):
        """Set GPU thread mode and count, and adjust dataset threads count."""
        cpu_count = multiprocessing.cpu_count()
        logging.info('Logical CPU cores: %s', cpu_count)

        per_gpu_thread_count = per_gpu_thread_count or 2
        #private_threads = (cpu_count -  strategy.num_replicas_in_sync * (per_gpu_thread_count + per_gpu_thread_count))
        num_runtime_threads = num_gpus

        total_gpu_thread_count = per_gpu_thread_count * num_gpus
        if not datasets_num_private_threads:
            datasets_num_private_threads = min(
                cpu_count - total_gpu_thread_count - num_runtime_threads,
                num_gpus * 8)
            logging.info('Set datasets_num_private_threads to %s',
                         datasets_num_private_threads)
        return datasets_num_private_threads

    def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                      datasets_num_private_threads,
                                      num_gpus, per_gpu_thread_count):

        # Allocate private thread pool for each GPU to schedule and launch kernels
        per_gpu_thread_count = per_gpu_thread_count or 2
        os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
        os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
        logging.info('TF_GPU_THREAD_COUNT: %s',
                     os.environ['TF_GPU_THREAD_COUNT'])
        logging.info('TF_GPU_THREAD_MODE: %s',
                     os.environ['TF_GPU_THREAD_MODE'])

    #         # Limit data preprocessing threadpool to CPU cores minus number of total GPU
    #         # private threads and memory copy threads.
    #         total_gpu_thread_count = per_gpu_thread_count * num_gpus
    #         num_runtime_threads = num_gpus
    #         if not datasets_num_private_threads:
    #             datasets_num_private_threads = min(
    #                 cpu_count - total_gpu_thread_count - num_runtime_threads,
    #                 num_gpus * 8)
    #             logging.info('Set datasets_num_private_threads to %s',
    #                          datasets_num_private_threads)

    def set_cudnn_batchnorm_mode(batchnorm_spatial_persistent):
        """Set CuDNN batchnorm mode for better performance.

           Note: Spatial Persistent mode may lead to accuracy losses for certain
           models.
        """
        if batchnorm_spatial_persistent:
            os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        else:
            os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)


    #################################################################

    #     data_dir = "hdfs:///Projects/demo_deep_learning_admin000/demo_deep_learning_admin000_Training_Datasets" #hdfs.project_path()
    #     train_filenames = pydoop.path.abspath(data_dir + "/cifar/train.tfrecord")
    #     train_filenames = tf.io.gfile.glob(train_filenames + "/part-r-*")
    #     test_filenames = pydoop.path.abspath(data_dir + "/cifar/test.tfrecord")
    #     test_filenames = tf.io.gfile.glob(test_filenames + "/part-r-*")

    train_path = "file:///staging/hopsworks-poc/spark/train_fake.tfrecord"
    eval_path = "file:///staging/hopsworks-poc/spark/eval_fake.tfrecord"
    train_filenames = tf.io.gfile.glob(train_path + "/part-r-*")
    test_filenames = tf.io.gfile.glob(eval_path + "/part-r-*")


    def _parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""

        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([height * width * channels], tf.float32),
            })

        image = features['image']
        label = features['label']

        return image, label

    def _reshape_img(image, label):
        image = tf.reshape(image, [height, width, channels])
        # label = tf.one_hot(label, num_classes)
        return image, label

    def _dataset_gen(tfrecord_files, height, width, channels, epochs, batch_size, steps_per_epoch, private_threads, strategy=None, train_mode = 'custom_loop'):
        dataset = tf.data.Dataset.list_files(tfrecord_files)
        dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls = tf.data.experimental.AUTOTUNE, cycle_length=tf.data.experimental.AUTOTUNE) #

        dataset = dataset.map(_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(_reshape_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.cache()

        dataset = dataset.shuffle(epochs * batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(epochs * steps_per_epoch)

        #dataset option
        options = tf.data.Options()
        options.experimental_optimization.map_vectorization.enabled = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.apply_default_optimizations = False
        if strategy:
            options.experimental_threading.private_threadpool_size = private_threads
        #            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        if strategy and train_mode == 'custom_loop':
            dataset = strategy.experimental_distribute_dataset(dataset)
        return dataset



    ######################################################################################################################
    data_format = None
    use_tf_while_loop = True
    enable_checkpoint_and_export = False
    enable_tensorboard = True
    skip_eval = False
    use_tf_function = True #'Wrap the train and test step inside a tf.function.'

    #--------------------------------------------------------------------------------------------------------------------
    #     enable_xla = False
    #     single_l2_loss_op = True #'Calculate L2_loss on concatenated weights, instead of using Keras per-layer L2 loss.'

    #     dtype= tf.float16
    #     fp16_implementation = 'keras'
    #     loss_scale =  'dynamic' #None
    #     num_packs = 8
    #     bytes_per_pack = 32 * 1024 * 1024  #None

    #     global_batch_size = 512

    #     train_epochs = 2
    #     epochs_between_evals = 1

    #     steps_per_loop = 50
    #     log_steps = 50


    keras_utils.set_session_config(
        enable_xla=enable_xla)
    performance.set_mixed_precision_policy(dtype)


    ######################################################################################################################

    per_gpu_thread_count = 2
    datasets_num_private_threads = None
    #     tf_gpu_thread_mode = 'gpu_private' #None
    #     batchnorm_spatial_persistent = True
    #     num_gpus = 8
    #     distribution_strategy = 'mirrored' #'multi_worker_mirrored'
    #     all_reduce_alg = 'nccl' #None

    datasets_num_private_threads = get_datasets_num_private_threads(datasets_num_private_threads, num_gpus, per_gpu_thread_count)


    #     if tf.config.list_physical_devices('GPU'):
    if tf_gpu_thread_mode:
        set_gpu_thread_mode_and_count(
            per_gpu_thread_count=per_gpu_thread_count,
            gpu_thread_mode=tf_gpu_thread_mode,
            num_gpus=num_gpus,
            datasets_num_private_threads=datasets_num_private_threads)
        set_cudnn_batchnorm_mode(batchnorm_spatial_persistent)

    # TODO: Set data_format without using Keras.
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

    ###################################
    num_images_train = 100000
    num_images_val = 20000

    # Image dimensions
    #     height = 32
    #     width = 32
    #     channels = 3

    height = 224
    width = 224
    channels = 3


    model_dir = tensorboard.logdir()

    #         if distribution_strategy == 'mirrored':
    #             BATCH_SIZE_PER_REPLICA = int(512 / strategy.num_replicas_in_sync)
    #         elif strategy_mode == 'multi_worker_mirrored':
    #             BATCH_SIZE_PER_REPLICA = int(1024 / strategy.num_replicas_in_sync)
    #         elif strategy_mode == None:
    #             BATCH_SIZE_PER_REPLICA = 512

    #per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations( flags_obj)
    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations_non_flag(global_batch_size, train_epochs,
                                                                                  num_images_train,
                                                                                  num_images_val,
                                                                                  provided_train_steps=None)

    train_dist_dataset = _dataset_gen(train_filenames, height, width, channels, train_epochs, global_batch_size, per_epoch_steps, datasets_num_private_threads, strategy)
    eval_dist_dataset = _dataset_gen(test_filenames, height, width, channels, train_epochs, global_batch_size, eval_steps, datasets_num_private_threads, strategy)


    steps_per_loop = min(steps_per_loop, per_epoch_steps)

    logging.info(
        'Training %d epochs, each epoch has %d steps, '
        'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
        train_epochs * per_epoch_steps, eval_steps)

    time_callback = keras_utils.TimeHistory(
        global_batch_size,
        log_steps,
        logdir=model_dir if enable_tensorboard else None)
    with distribution_utils.get_strategy_scope(strategy):
        runnable = resnet_hops_runnable.ResnetHopsRunnable(train_dist_dataset, eval_dist_dataset,
                                                           use_tf_while_loop, use_tf_function, dtype, global_batch_size,
                                                           datasets_num_private_threads, single_l2_loss_op, loss_scale,
                                                           fp16_implementation, bytes_per_pack, num_images_train,
                                                           time_callback, per_epoch_steps)

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

    # https://github.com/tensorflow/tensorflow/releases
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
    # https://gist.github.com/ceshine/0549086d8c59efb1d706f6e369b8e136#file-profiling-ipynb
    # https://github.com/ceshine/tf-helper-bot/tree/master/tf_helper_bot
    #WARNING: Profile doesnt work on custom loop: https://github.com/tensorflow/tensorboard/issues/3288
    #with tf.profiler.experimental.Profile(model_dir):
    time_callback.on_train_begin()
    # Train the model here
    resnet_controller.train(evaluate=not skip_eval)
    time_callback.on_train_end()


    stats = build_stats(runnable, skip_eval, time_callback)
    return stats