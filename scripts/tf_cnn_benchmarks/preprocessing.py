# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Image pre-processing utilities.
"""

import math
import tensorflow as tf
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


_RESIZE_METHOD_MAP = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}


def get_image_resize_method(resize_method, batch_position=0):
    """Get tensorflow resize method.

    Args:
      resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
      batch_position: position of the image in a batch. This can be an integer or a tensor.
    Returns:
      one of resize type defined in tf.image.ResizeMethod.
    """
    if resize_method != 'round_robin':
        return _RESIZE_METHOD_MAP[resize_method]

    # Return a resize method based on batch position in a round-robin fashion.
    resize_methods = list(_RESIZE_METHOD_MAP.values())

    def lookup(index):
        return resize_methods[index]

    def resize_method_0():
        return smart_cond.smart_cond(batch_position % len(resize_methods) == 0,
                                     lambda: lookup(0), resize_method_1)

    def resize_method_1():
        return smart_cond.smart_cond(batch_position % len(resize_methods) == 1,
                                     lambda: lookup(1), resize_method_2)

    def resize_method_2():
        return smart_cond.smart_cond(batch_position % len(resize_methods) == 2,
                                     lambda: lookup(2), lambda: lookup(3))

    return resize_method_0()


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope or 'decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        return image


def normalized_image(images):
    # Rescale from [0, 255] to [0, 2]
    images = tf.multiply(images, 1. / 127.5)
    # Rescale to [-1, 1]
    return tf.subtract(images, 1.0)


def eval_image(image,
               height,
               width,
               batch_position,
               resize_method,
               summary_verbosity=0):
    """Get the image for model evaluation.

    Args:
      image: 3-D float Tensor representing the image.
      height: The height of the image that will be returned.
      width: The width of the image that will be returned.
      batch_position: position of the image in a batch. This can be an integer or a tensor.
      resize_method: one of the strings 'round_robin', 'nearest', 'bilinear', 'bicubic', or 'area'.
      summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both summaries and checkpoints.
    Returns:
      An image of size (output_height, output_width, 3) that is resized and cropped as described above.
    """
    with tf.name_scope('eval_image'):
        if summary_verbosity >= 3:
            tf.summary.image('original_image', tf.expand_dims(image, 0))

        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]
        image_height_float = tf.cast(image_height, tf.float32)
        image_width_float = tf.cast(image_width, tf.float32)

        scale_factor = 1.15

        max_ratio = tf.maximum(height / image_height_float,
                               width / image_width_float)
        resize_height = tf.cast(image_height_float * max_ratio * scale_factor,
                                tf.int32)
        resize_width = tf.cast(image_width_float * max_ratio * scale_factor,
                               tf.int32)

        image_resize_method = get_image_resize_method(resize_method, batch_position)
        distorted_image = tf.image.resize(image,
                                          [resize_height, resize_width],
                                          method=image_resize_method)

        total_crop_height = (resize_height - height)
        crop_top = total_crop_height // 2
        total_crop_width = (resize_width - width)
        crop_left = total_crop_width // 2
        distorted_image = tf.image.crop_to_bounding_box(distorted_image, crop_top, crop_left, height, width)

        distorted_image.set_shape([height, width, 3])
        if summary_verbosity >= 3:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        image = distorted_image
    return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
    """Distort one image for training a network.

    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as [ymin, xmin, ymax, xmax].
      batch_position: position of the image in a batch. This can be an integer or a tensor.
      resize_method: round_robin, nearest, bilinear, bicubic, or area.
      distortions: If true, apply full distortions for image colors.
      scope: Optional scope for op_scope.
      summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both summaries and checkpoints.
      distort_color_in_yiq: distort color of input images in YIQ space.
      fuse_decode_and_crop: fuse the decode/crop operation.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(scope or 'distort_image'):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if summary_verbosity >= 3:
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)

        if fuse_decode_and_crop:
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.decode_and_crop_jpeg(
                image_buffer, crop_window, channels=3)
        else:
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = tf.slice(image, bbox_begin, bbox_size)

        distorted_image = tf.image.random_flip_left_right(image)

        image_resize_method = get_image_resize_method(resize_method, batch_position)
        distorted_image = tf.image.resize(distorted_image, [height, width], method=image_resize_method)

        distorted_image.set_shape([height, width, 3])
        if summary_verbosity >= 3:
            tf.summary.image('cropped_resized_maybe_flipped_image', tf.expand_dims(distorted_image, 0))

        if distortions:
            distorted_image = tf.cast(distorted_image, dtype=tf.float32)
            distorted_image /= 255.
            distorted_image = distort_color(distorted_image, batch_position, distort_color_in_yiq=distort_color_in_yiq)
            distorted_image *= 255

        if summary_verbosity >= 3:
            tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
        return distorted_image


def distort_color(image, batch_position=0, distort_color_in_yiq=False, scope=None):
    """Distort the color of the image.

    Args:
      image: float32 Tensor containing single image. Tensor values should be in range [0, 1].
      batch_position: the position of the image in a batch. This can be an integer or a tensor.
      distort_color_in_yiq: distort color of input images in YIQ space.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(scope or 'distort_color'):

        def distort_fn_0(image=image):
            """Variant 0 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            if distort_color_in_yiq:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def distort_fn_1(image=image):
            """Variant 1 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            if distort_color_in_yiq:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            return image

        image = smart_cond.smart_cond(batch_position % 2 == 0, distort_fn_0, distort_fn_1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

# 继续定义其他类和函数，确保使用的 API 符合 TensorFlow 2.5.0 的规范。

class InputPreprocessor(object):
    """Base class for all model preprocessors."""

    def __init__(self, batch_size, output_shapes):
        self.batch_size = batch_size
        self.output_shapes = output_shapes

    def supports_datasets(self):
        """Whether this preprocessor supports dataset."""
        return False

    def minibatch(self, dataset, subset, params, shift_ratio=-1):
        """Returns tensors representing a minibatch of all the input."""
        raise NotImplementedError('Must be implemented by subclass.')

    def parse_and_preprocess(self, value, batch_position):
        """Function to parse and preprocess an Example proto in input pipeline."""
        raise NotImplementedError('Must be implemented by subclass.')

    def build_prefetch_input_processing(self, batch_size, model_input_shapes,
                                        num_splits, cpu_device, params,
                                        gpu_devices, model_input_data_types,
                                        dataset, doing_eval):
        """Returns FunctionBufferingResources that do input pre(processing)."""
        assert self.supports_datasets()
        with tf.device(cpu_device):
            subset = 'validation' if doing_eval else 'train'

            function_buffering_resources = []
            remote_fn, args = self.minibatch_fn(
                batch_size=batch_size,
                model_input_shapes=model_input_shapes,
                num_splits=num_splits,
                dataset=dataset,
                subset=subset,
                train=(not doing_eval),
                datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
                num_threads=params.datasets_num_private_threads,
                datasets_use_caching=params.datasets_use_caching,
                datasets_parallel_interleave_cycle_length=(
                    params.datasets_parallel_interleave_cycle_length),
                datasets_sloppy_parallel_interleave=(
                    params.datasets_sloppy_parallel_interleave),
                datasets_parallel_interleave_prefetch=(
                    params.datasets_parallel_interleave_prefetch))
            for device_num in range(len(gpu_devices)):
                with tf.device(gpu_devices[device_num]):
                    buffer_resource_handle = tf.raw_ops.FunctionBufferingResource(
                        f=remote_fn,
                        output_types=model_input_data_types,
                        target_device=cpu_device,
                        string_arg=args[0],
                        buffer_size=params.datasets_prefetch_buffer_size,
                        shared_name=None)
                    function_buffering_resources.append(buffer_resource_handle)
            return function_buffering_resources

    def build_multi_device_iterator(self, batch_size, num_splits, cpu_device,
                                    params, gpu_devices, dataset, doing_eval):
        """Creates a MultiDeviceIterator."""
        assert self.supports_datasets()
        assert num_splits == len(gpu_devices)
        with tf.name_scope('batch_processing'):
            subset = 'validation' if doing_eval else 'train'
            batch_size_per_split = batch_size // num_splits
            ds = self.create_dataset(
                batch_size,
                num_splits,
                batch_size_per_split,
                dataset,
                subset,
                train=(not doing_eval),
                datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
                num_threads=params.datasets_num_private_threads,
                datasets_use_caching=params.datasets_use_caching,
                datasets_parallel_interleave_cycle_length=(
                    params.datasets_parallel_interleave_cycle_length),
                datasets_sloppy_parallel_interleave=(
                    params.datasets_sloppy_parallel_interleave),
                datasets_parallel_interleave_prefetch=(
                    params.datasets_parallel_interleave_prefetch))
            multi_device_iterator = tf.data.experimental.MultiDeviceIterator(
                ds,
                gpu_devices,
                source_device=cpu_device,
                max_buffer_size=params.multi_device_iterator_max_buffer_size)
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS,
                                           multi_device_iterator.initializer)
            return multi_device_iterator

    def create_dataset(self,
                       batch_size,
                       num_splits,
                       batch_size_per_split,
                       dataset,
                       subset,
                       train,
                       datasets_repeat_cached_sample,
                       num_threads=None,
                       datasets_use_caching=False,
                       datasets_parallel_interleave_cycle_length=None,
                       datasets_sloppy_parallel_interleave=False,
                       datasets_parallel_interleave_prefetch=None):
        """Creates a dataset for the benchmark."""
        raise NotImplementedError('Must be implemented by subclass.')

    def create_iterator(self, ds):
        ds_iterator = iter(ds)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS,
                                       ds_iterator.initializer)
        return ds_iterator

    def minibatch_fn(self, batch_size, model_input_shapes, num_splits,
                     dataset, subset, train, datasets_repeat_cached_sample,
                     num_threads, datasets_use_caching,
                     datasets_parallel_interleave_cycle_length,
                     datasets_sloppy_parallel_interleave,
                     datasets_parallel_interleave_prefetch):
        """Returns a function and list of args for the fn to create a minibatch."""
        assert self.supports_datasets()
        batch_size_per_split = batch_size // num_splits
        assert batch_size_per_split == model_input_shapes[0][0]
        with tf.name_scope('batch_processing'):
            ds = self.create_dataset(batch_size, num_splits, batch_size_per_split,
                                     dataset, subset, train,
                                     datasets_repeat_cached_sample, num_threads,
                                     datasets_use_caching,
                                     datasets_parallel_interleave_cycle_length,
                                     datasets_sloppy_parallel_interleave,
                                     datasets_parallel_interleave_prefetch)
            ds_iterator = self.create_iterator(ds)

            ds_iterator_string_handle = ds_iterator.string_handle()

            @tf.function
            def _fn(h):
                remote_iterator = tf.data.Iterator.from_string_handle(
                    h, ds_iterator.output_types, ds_iterator.output_shapes)
                input_list = remote_iterator.get_next()
                reshaped_input_list = [
                    tf.reshape(input_list[i], shape=model_input_shapes[i])
                    for i in range(len(input_list))
                ]
                return reshaped_input_list

            return _fn, [ds_iterator_string_handle]


class BaseImagePreprocessor(InputPreprocessor):
    """Base class for all image model preprocessors."""

    def __init__(self,
                 batch_size,
                 output_shapes,
                 num_splits,
                 dtype,
                 train,
                 distortions,
                 resize_method,
                 shift_ratio=-1,
                 summary_verbosity=0,
                 distort_color_in_yiq=True,
                 fuse_decode_and_crop=True):
        super(BaseImagePreprocessor, self).__init__(batch_size, output_shapes)
        image_shape = output_shapes[0]
        # image_shape is in form (batch_size, height, width, depth)
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.depth = image_shape[3]
        self.num_splits = num_splits
        self.dtype = dtype
        self.train = train
        self.resize_method = resize_method
        self.shift_ratio = shift_ratio
        self.distortions = distortions
        self.distort_color_in_yiq = distort_color_in_yiq
        self.fuse_decode_and_crop = fuse_decode_and_crop
        if self.batch_size % self.num_splits != 0:
            raise ValueError(
                ('batch_size must be a multiple of num_splits: '
                 'batch_size %d, num_splits: %d') %
                (self.batch_size, self.num_splits))
        self.batch_size_per_split = self.batch_size // self.num_splits
        self.summary_verbosity = summary_verbosity

    def parse_and_preprocess(self, value, batch_position):
        assert self.supports_datasets()
        image_buffer, label_index, bbox, _ = parse_example_proto(value)
        image = self.preprocess(image_buffer, bbox, batch_position)
        return (image, label_index)

    def preprocess(self, image_buffer, bbox, batch_position):
        raise NotImplementedError('Must be implemented by subclass.')

    def create_dataset(self,
                       batch_size,
                       num_splits,
                       batch_size_per_split,
                       dataset,
                       subset,
                       train,
                       datasets_repeat_cached_sample,
                       num_threads=None,
                       datasets_use_caching=False,
                       datasets_parallel_interleave_cycle_length=None,
                       datasets_sloppy_parallel_interleave=False,
                       datasets_parallel_interleave_prefetch=None):
        """Creates a dataset for the benchmark."""
        assert self.supports_datasets()
        glob_pattern = dataset.tf_record_pattern(subset)
        file_names = tf.io.gfile.glob(glob_pattern)
        if not file_names:
            raise ValueError('Found no files in --data_dir matching: {}'
                             .format(glob_pattern))
        ds = tf.data.Dataset.list_files(file_names)
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            num_parallel_calls=tf.data.AUTOTUNE if datasets_sloppy_parallel_interleave else None)
        if datasets_repeat_cached_sample:
            # Repeat a single sample element indefinitely to emulate memory-speed IO.
            ds = ds.take(1).cache().repeat()
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        ds = tf.data.Dataset.zip((ds, counter))
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if datasets_use_caching:
            ds = ds.cache()
        if train:
            ds = ds.shuffle(buffer_size=10000).repeat()
        else:
            ds = ds.repeat()
        ds = ds.map(
            self.parse_and_preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size_per_split, drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


class RecordInputImagePreprocessor(BaseImagePreprocessor):
    """Preprocessor for images with RecordInput format."""

    def preprocess(self, image_buffer, bbox, batch_position):
        """Preprocessing image_buffer as a function of its batch position."""
        if self.train:
            image = train_image(image_buffer, self.height, self.width, bbox,
                                batch_position, self.resize_method, self.distortions,
                                None, summary_verbosity=self.summary_verbosity,
                                distort_color_in_yiq=self.distort_color_in_yiq,
                                fuse_decode_and_crop=self.fuse_decode_and_crop)
        else:
            image = tf.image.decode_jpeg(
                image_buffer, channels=3)
            image = eval_image(image, self.height, self.width, batch_position,
                               self.resize_method,
                               summary_verbosity=self.summary_verbosity)
        # Note: image is now float32 [height,width,3] with range [0, 255]

        normalized = normalized_image(image)
        return tf.cast(normalized, self.dtype)

    def minibatch(self,
                  dataset,
                  subset,
                  params,
                  shift_ratio=-1):
        if shift_ratio < 0:
            shift_ratio = self.shift_ratio
        with tf.name_scope('batch_processing'):
            images = [[] for _ in range(self.num_splits)]
            labels = [[] for _ in range(self.num_splits)]
            if params.use_datasets:
                ds = self.create_dataset(
                    self.batch_size, self.num_splits, self.batch_size_per_split,
                    dataset, subset, self.train,
                    datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
                    num_threads=params.datasets_num_private_threads,
                    datasets_use_caching=params.datasets_use_caching,
                    datasets_parallel_interleave_cycle_length=(
                        params.datasets_parallel_interleave_cycle_length),
                    datasets_sloppy_parallel_interleave=(
                        params.datasets_sloppy_parallel_interleave),
                    datasets_parallel_interleave_prefetch=(
                        params.datasets_parallel_interleave_prefetch))
                ds_iterator = iter(ds)
                for d in range(self.num_splits):
                    images[d], labels[d] = ds_iterator.get_next()

            else:
                record_input = data_flow_ops.RecordInput(
                    file_pattern=dataset.tf_record_pattern(subset),
                    seed=301,
                    parallelism=64,
                    buffer_size=10000,
                    batch_size=self.batch_size,
                    shift_ratio=shift_ratio,
                    name='record_input')
                records = record_input.get_yield_op()
                records = tf.split(records, self.batch_size, 0)
                records = [tf.reshape(record, []) for record in records]
                for idx in range(self.batch_size):
                    value = records[idx]
                    (image, label) = self.parse_and_preprocess(value, idx)
                    split_index = idx % self.num_splits
                    labels[split_index].append(label)
                    images[split_index].append(image)

            for split_index in range(self.num_splits):
                if not params.use_datasets:
                    images[split_index] = tf.stack(images[split_index])
                    labels[split_index] = tf.concat(labels[split_index], 0)
                images[split_index] = tf.reshape(
                    images[split_index],
                    shape=[self.batch_size_per_split, self.height, self.width,
                           self.depth])
                labels[split_index] = tf.reshape(labels[split_index],
                                                 [self.batch_size_per_split])
            return images, labels

    def supports_datasets(self):
        return True


class ImagenetPreprocessor(RecordInputImagePreprocessor):

    def preprocess(self, image_buffer, bbox, batch_position):
        try:
            from official.vision.image_classification.resnet.imagenet_preprocessing import preprocess_image
        except ImportError:
            raise ImportError('Please include tensorflow/models to the PYTHONPATH.')
        if self.train:
            image = preprocess_image(
                image_buffer, bbox, self.height, self.width, self.depth,
                is_training=True)
        else:
            image = preprocess_image(
                image_buffer, bbox, self.height, self.width, self.depth,
                is_training=False)
        return tf.cast(image, self.dtype)


class Cifar10ImagePreprocessor(BaseImagePreprocessor):
    """Preprocessor for Cifar10 input images."""

    def _distort_image(self, image):
        """Distort one image for training a network.

        Adopted the standard data augmentation scheme that is widely used for
        this dataset: the images are first zero-padded with 4 pixels on each side,
        then randomly cropped to again produce distorted images; half of the images
        are then horizontally mirrored.

        Args:
          image: input image.
        Returns:
          distorted image.
        """
        image = tf.image.resize_with_crop_or_pad(
            image, self.height + 8, self.width + 8)
        distorted_image = tf.image.random_crop(image,
                                               [self.height, self.width, self.depth])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if self.summary_verbosity >= 3:
            tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
        return distorted_image

    def _eval_image(self, image):
        """Get the image for model evaluation."""
        distorted_image = tf.image.resize_with_crop_or_pad(
            image, self.width, self.height)
        if self.summary_verbosity >= 3:
            tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
        return distorted_image

    def preprocess(self, raw_image):
        """Preprocessing raw image."""
        if self.summary_verbosity >= 3:
            tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
        if self.train and self.distortions:
            image = self._distort_image(raw_image)
        else:
            image = self._eval_image(raw_image)
        normalized = normalized_image(image)
        return tf.cast(normalized, self.dtype)

    def minibatch(self,
                  dataset,
                  subset,
                  params,
                  shift_ratio=-1):
        del shift_ratio, params
        with tf.name_scope('batch_processing'):
            all_images, all_labels = dataset.read_data_files(subset)
            all_images = tf.constant(all_images)
            all_labels = tf.constant(all_labels)
            input_image, input_label = tf.compat.v1.train.slice_input_producer(
                [all_images, all_labels])
            input_image = tf.cast(input_image, self.dtype)
            input_label = tf.cast(input_label, tf.int32)
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                                     min_fraction_of_examples_in_queue)
            raw_images, raw_labels = tf.compat.v1.train.shuffle_batch(
                [input_image, input_label], batch_size=self.batch_size,
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=min_queue_examples)

            images = [[] for i in range(self.num_splits)]
            labels = [[] for i in range(self.num_splits)]

            raw_images = tf.unstack(raw_images, axis=0)
            raw_labels = tf.unstack(raw_labels, axis=0)
            for i in range(self.batch_size):
                split_index = i % self.num_splits
                raw_image = tf.reshape(raw_images[i],
                                       [dataset.depth, dataset.height, dataset.width])
                raw_image = tf.transpose(raw_image, [1, 2, 0])
                image = self.preprocess(raw_image)
                images[split_index].append(image)

                labels[split_index].append(raw_labels[i])

            for split_index in range(self.num_splits):
                images[split_index] = tf.stack(images[split_index])
                labels[split_index] = tf.stack(labels[split_index])
            return images, labels


# 继续定义其他类和函数，确保使用的 API 符合 TensorFlow 2.5.0 的规范。

class COCOPreprocessor(BaseImagePreprocessor):
    """Preprocessor for COCO dataset input images, boxes, and labels."""

    def minibatch(self,
                  dataset,
                  subset,
                  params,
                  shift_ratio=-1):
        del shift_ratio  # Not used when using datasets instead of data_flow_ops
        with tf.name_scope('batch_processing'):
            ds = self.create_dataset(
                self.batch_size, self.num_splits, self.batch_size_per_split,
                dataset, subset, self.train, params.datasets_repeat_cached_sample)
            ds_iterator = iter(ds)

            # Training data: 4 tuple
            # Validation data: 5 tuple
            # See get_input_shapes in models/ssd_model.py for details.
            input_len = 4 if subset == 'train' else 5
            input_lists = [[None for _ in range(self.num_splits)]
                           for _ in range(input_len)]
            for d in range(self.num_splits):
                input_list = next(ds_iterator)
                for i in range(input_len):
                    input_lists[i][d] = input_list[i]
            return input_lists

    def preprocess(self, data):
        try:
            import ssd_dataloader  # Assuming this module is available
            import ssd_constants  # Assuming this module is available
            from object_detection.core import preprocessor  # Assuming this module is available
        except ImportError:
            raise ImportError('To use the COCO dataset, you must clone the '
                              'repo https://github.com/tensorflow/models and add '
                              'tensorflow/models and tensorflow/models/research to '
                              'the PYTHONPATH, and compile the protobufs by '
                              'following the installation guide.')

        source_id = tf.strings.to_number(data['source_id'], out_type=tf.int32)
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        raw_shape = tf.shape(image)
        boxes = data['groundtruth_boxes']
        classes = tf.reshape(data['groundtruth_classes'], [-1, 1])

        # Only 80 of the 90 COCO classes are used.
        class_map = tf.convert_to_tensor(ssd_constants.CLASS_MAP)
        classes = tf.gather(class_map, classes)
        classes = tf.cast(classes, dtype=tf.float32)

        if self.train:
            image, boxes, classes = ssd_dataloader.ssd_crop(image, boxes, classes)
            image, boxes = preprocessor.random_horizontal_flip(
                image=image, boxes=boxes)

            image = ssd_dataloader.color_jitter(
                image, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
            image = ssd_dataloader.normalize_image(image)
            image = tf.cast(image, self.dtype)

            encoded_returns = ssd_dataloader.encode_labels(boxes, classes)
            encoded_classes, encoded_boxes, num_matched_boxes = encoded_returns

            # Shape of image: [width, height, channel]
            # Shape of encoded_boxes: [NUM_SSD_BOXES, 4]
            # Shape of encoded_classes: [NUM_SSD_BOXES, 1]
            # Shape of num_matched_boxes: [1]
            return (image, encoded_boxes, encoded_classes, num_matched_boxes)

        else:
            image = tf.image.resize(
                image[tf.newaxis, :, :, :],
                size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE)
            )[0, :, :, :]

            image = ssd_dataloader.normalize_image(image)
            image = tf.cast(image, self.dtype)

            def trim_and_pad(inp_tensor):
                """Limit the number of boxes, and pad if necessary."""
                inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
                num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
                inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
                return tf.reshape(inp_tensor, [ssd_constants.MAX_NUM_EVAL_BOXES,
                                               inp_tensor.get_shape()[1]])

            boxes, classes = trim_and_pad(boxes), trim_and_pad(classes)

            # Shape of boxes: [MAX_NUM_EVAL_BOXES, 4]
            # Shape of classes: [MAX_NUM_EVAL_BOXES, 1]
            # Shape of source_id: [] (scalar tensor)
            # Shape of raw_shape: [3]
            return (image, boxes, classes, source_id, raw_shape)

    def create_dataset(self,
                       batch_size,
                       num_splits,
                       batch_size_per_split,
                       dataset,
                       subset,
                       train,
                       datasets_repeat_cached_sample,
                       num_threads=None,
                       datasets_use_caching=False,
                       datasets_parallel_interleave_cycle_length=None,
                       datasets_sloppy_parallel_interleave=False,
                       datasets_parallel_interleave_prefetch=None):
        """Creates a dataset for the benchmark."""
        try:
            from object_detection.data_decoders import tf_example_decoder  # Assuming this module is available
        except ImportError:
            raise ImportError('To use the COCO dataset, you must clone the '
                              'repo https://github.com/tensorflow/models and add '
                              'tensorflow/models and tensorflow/models/research to '
                              'the PYTHONPATH, and compile the protobufs by '
                              'following the installation guide.')

        assert self.supports_datasets()
        example_decoder = tf_example_decoder.TfExampleDecoder()

        glob_pattern = dataset.tf_record_pattern(subset)
        file_names = tf.io.gfile.glob(glob_pattern)
        if not file_names:
            raise ValueError('Found no files in --data_dir matching: {}'
                             .format(glob_pattern))

        ds = tf.data.Dataset.list_files(file_names)
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            num_parallel_calls=tf.data.AUTOTUNE if datasets_sloppy_parallel_interleave else None)
        if datasets_repeat_cached_sample:
            # Repeat a single sample element indefinitely to emulate memory-speed IO.
            ds = ds.take(1).cache().repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if datasets_use_caching:
            ds = ds.cache()
        if train:
            ds = ds.shuffle(buffer_size=10000).repeat()

        ds = ds.map(example_decoder.decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(
            lambda data: tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0))
        ds = ds.map(
            self.preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size_per_split, drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def supports_datasets(self):
        return True


class TestImagePreprocessor(BaseImagePreprocessor):
    """Preprocessor used for testing.

    set_fake_data() sets which images and labels will be output by minibatch(),
    and must be called before minibatch(). This allows tests to easily specify
    a set of images to use for training, without having to create any files.

    Queue runners must be started for this preprocessor to work.
    """

    def __init__(self,
                 batch_size,
                 output_shapes,
                 num_splits,
                 dtype,
                 train=None,
                 distortions=None,
                 resize_method=None,
                 shift_ratio=0,
                 summary_verbosity=0,
                 distort_color_in_yiq=False,
                 fuse_decode_and_crop=False):
        super(TestImagePreprocessor, self).__init__(
            batch_size, output_shapes, num_splits, dtype, train, distortions,
            resize_method, shift_ratio, summary_verbosity=summary_verbosity,
            distort_color_in_yiq=distort_color_in_yiq,
            fuse_decode_and_crop=fuse_decode_and_crop)
        self.expected_subset = None

    def set_fake_data(self, fake_images, fake_labels):
        assert len(fake_images.shape) == 4
        assert len(fake_labels.shape) == 1
        num_images = fake_images.shape[0]
        assert num_images == fake_labels.shape[0]
        assert num_images % self.batch_size == 0
        self.fake_images = fake_images
        self.fake_labels = fake_labels

    def minibatch(self,
                  dataset,
                  subset,
                  params,
                  shift_ratio=0):
        """Get test image batches."""
        del dataset, params
        if (not hasattr(self, 'fake_images') or
                not hasattr(self, 'fake_labels')):
            raise ValueError('Must call set_fake_data() before calling minibatch '
                             'on TestImagePreprocessor')
        if self.expected_subset is not None:
            assert subset == self.expected_subset

        shift_ratio = shift_ratio or self.shift_ratio
        fake_images = cnn_util.roll_numpy_batches(self.fake_images, self.batch_size,
                                                  shift_ratio)
        fake_labels = cnn_util.roll_numpy_batches(self.fake_labels, self.batch_size,
                                                  shift_ratio)

        with tf.name_scope('batch_processing'):
            image_slice, label_slice = tf.compat.v1.train.slice_input_producer(
                [fake_images, fake_labels],
                shuffle=False,
                name='image_slice')
            raw_images, raw_labels = tf.compat.v1.train.batch(
                [image_slice, label_slice], batch_size=self.batch_size,
                name='image_batch')
            images = [[] for _ in range(self.num_splits)]
            labels = [[] for _ in range(self.num_splits)]
            for i in range(self.batch_size):
                split_index = i % self.num_splits
                raw_image = tf.cast(raw_images[i], self.dtype)
                images[split_index].append(raw_image)
                labels[split_index].append(raw_labels[i])
            for split_index in range(self.num_splits):
                images[split_index] = tf.stack(images[split_index])
                labels[split_index] = tf.stack(labels[split_index])

            normalized = [normalized_image(part) for part in images]
            return [[tf.cast(part, self.dtype) for part in normalized], labels]


class LibrispeechPreprocessor(InputPreprocessor):
    """Preprocessor for Librispeech dataset."""

    def __init__(self, batch_size, output_shapes, num_splits, dtype, train,
                 **kwargs):
        del kwargs
        super(LibrispeechPreprocessor, self).__init__(batch_size, output_shapes)
        self.num_splits = num_splits
        self.dtype = dtype
        self.is_train = train
        if self.batch_size % self.num_splits != 0:
            raise ValueError(('batch_size must be a multiple of num_splits: '
                              'batch_size %d, num_splits: %d') % (self.batch_size,
                                                                  self.num_splits))
        self.batch_size_per_split = self.batch_size // self.num_splits

    def create_dataset(self,
                       batch_size,
                       num_splits,
                       batch_size_per_split,
                       dataset,
                       subset,
                       train,
                       datasets_repeat_cached_sample,
                       num_threads=None,
                       datasets_use_caching=False,
                       datasets_parallel_interleave_cycle_length=None,
                       datasets_sloppy_parallel_interleave=False,
                       datasets_parallel_interleave_prefetch=None):
        """Creates a dataset for the benchmark."""
        assert self.supports_datasets()
        glob_pattern = dataset.tf_record_pattern(subset)
        file_names = tf.io.gfile.glob(glob_pattern)
        if not file_names:
            raise ValueError('Found no files in --data_dir matching: {}'
                             .format(glob_pattern))
        ds = tf.data.Dataset.list_files(file_names)
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            num_parallel_calls=tf.data.AUTOTUNE if datasets_sloppy_parallel_interleave else None)
        if datasets_repeat_cached_sample:
            # Repeat a single sample element indefinitely to emulate memory-speed IO.
            ds = ds.take(1).cache().repeat()
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        ds = tf.data.Dataset.zip((ds, counter))
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if datasets_use_caching:
            ds = ds.cache()
        if train:
            ds = ds.shuffle(buffer_size=10000).repeat()
        ds = ds.map(map_func=self.parse_and_preprocess,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.padded_batch(
            batch_size=batch_size_per_split,
            padded_shapes=tuple([
                tf.TensorShape(output_shape[1:])
                for output_shape in self.output_shapes
            ]),
            drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def minibatch(self, dataset, subset, params, shift_ratio=-1):
        assert params.use_datasets
        del shift_ratio
        with tf.name_scope('batch_processing'):
            ds = self.create_dataset(
                self.batch_size,
                self.num_splits,
                self.batch_size_per_split,
                dataset,
                subset,
                self.is_train,
                datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
                num_threads=params.datasets_num_private_threads,
                datasets_use_caching=params.datasets_use_caching,
                datasets_parallel_interleave_cycle_length=(
                    params.datasets_parallel_interleave_cycle_length),
                datasets_sloppy_parallel_interleave=(
                    params.datasets_sloppy_parallel_interleave),
                datasets_parallel_interleave_prefetch=(
                    params.datasets_parallel_interleave_prefetch))
            ds_iterator = iter(ds)

            # The four lists are: input spectrogram feature, labels, input lengths,
            # label lengths
            input_lists = [[None for _ in range(self.num_splits)] for _ in range(4)]
            for d in range(self.num_splits):
                input_list = next(ds_iterator)
                for i in range(4):
                    input_lists[i][d] = input_list[i]

            assert self.output_shapes == [
                input_lists[i][0].shape.as_list() for i in range(4)
            ]
            return tuple(input_lists)

    def supports_datasets(self):
        return True

    def parse_and_preprocess(self, value, batch_position):
        """Parse a TFRecord."""
        del batch_position
        assert self.supports_datasets()
        context_features = {
            'labels': tf.io.VarLenFeature(dtype=tf.int64),
            'input_length': tf.io.FixedLenFeature([], dtype=tf.int64),
            'label_length': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        sequence_features = {
            'features': tf.io.FixedLenSequenceFeature([161], dtype=tf.float32)
        }
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized=value,
            context_features=context_features,
            sequence_features=sequence_features,
        )

        return [
            # Input
            tf.expand_dims(sequence_parsed['features'], axis=2),
            # Label
            tf.cast(
                tf.reshape(
                    tf.sparse.to_dense(context_parsed['labels']), [-1]),
                dtype=tf.int32),
            # Input length
            tf.cast(
                tf.reshape(context_parsed['input_length'], [1]),
                dtype=tf.int32),
            # Label length
            tf.cast(
                tf.reshape(context_parsed['label_length'], [1]),
                dtype=tf.int32),
        ]