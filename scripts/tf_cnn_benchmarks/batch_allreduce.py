# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import abc
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
import allreduce
import constants


def _all_reduce_using_copy(tensors_across_devices, use_mean):
    """Does an all-reduce of a list of tensors by copying to the current device."""
    reduced_tensor = tf.add_n(tensors_across_devices)
    if use_mean:
        reduced_tensor *= 1 / len(tensors_across_devices)
    return reduced_tensor


class BatchAllReduceAlgorithm(abc.ABC):
    """Represents an algorithm for performing a batch all-reduce operation."""

    def batch_all_reduce(self,
                         all_device_tensors,
                         num_splits,
                         compact_tensors,
                         defer_tensors,
                         xla_compile=False):
        """Performs a batch all-reduce.

        The reduction done is a sum.

        `all_device_tensors` is a list of list of tensors that will be batch
        all-reduced. All tensors within a single inner list must be on the same
        device. The nth element in each list, for any n, will be reduced together.
        The return value is in the same form as `all_device_tensors`, except that
        each tensor is reduced.

        Arguments:
          all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
            is a tensor where `i` is the device index and `j` is the tensor index.
          num_splits: If not None, tensors will be concatenated and split into this
            many pieces during the all-reduce, then split back into their original
            shapes afterwards. Has no impact on correctness and can improve
            performance. Requires all tensors to be the same type.
          compact_tensors: If True, tensors are casted to fp16 before being all-
            reduced. Improves performance, but hurts numerical stability.
          defer_tensors: If True, every time the return value
            `reduced_all_device_tensors` is evaluated, the result will be the
            reduced tensors values of `all_device_tensors` from the previous session
            run instead of the current session run, or zero on the first session
            run. This can improve performance. When training neural networks,
            deferring gradients often does not harm training, so this can be used to
            improve performance.
          xla_compile: If True, use XLA to compile gradients packing and unpacking
            ops.

        Returns:
          reduced_all_device_tensors: A list in the same form as
            `all_device_tensors`, except each tensor has been reduced.
          warmup_ops: A list of ops needed to be run once before the all-reduce can
            occur.
        """
        all_device_packed_tensors = []
        all_device_warmup_ops = []
        all_device_put_ops = []

        packers = [
            _TensorPacker(num_splits, compact_tensors) for _ in all_device_tensors
        ]

        for packer, device_tensors in zip(packers, all_device_tensors):

            def pack_single_device_tensors(packer=packer,
                                           device_tensors=device_tensors):
                """Pack gradient tensors of a device."""
                packed_tensors = packer.maybe_concat_tensors(device_tensors)
                packed_tensors = packer.maybe_compact_tensors(packed_tensors)
                if defer_tensors and not xla_compile:
                    packed_tensors, put_ops, warmup_ops = defer_single_device_tensors(
                        packed_tensors)
                    all_device_put_ops.append(put_ops)
                    all_device_warmup_ops.append(warmup_ops)
                packed_tensors = packer.maybe_split_tensors(packed_tensors)
                return packed_tensors

            with tf.device(device_tensors[0].device):
                if xla_compile:
                    packed_tensors = tf.function(pack_single_device_tensors, jit_compile=True)()
                    if defer_tensors:
                        packed_tensors, put_ops, warmup_ops = defer_single_device_tensors(
                            packed_tensors)
                        all_device_put_ops.append(put_ops)
                        all_device_warmup_ops.append(warmup_ops)
                else:
                    packed_tensors = pack_single_device_tensors()

            all_device_packed_tensors.append(packed_tensors)

        all_device_tensors = self._do_batch_all_reduce(all_device_packed_tensors)

        all_device_unpacked_tensors = []
        for packer, device_tensors in zip(packers, all_device_tensors):

            def unpack_single_device_tensors(packer=packer,
                                             device_tensors=device_tensors):
                """Unpack gradient tensors of a device."""
                unpacked_tensors = packer.undo_maybe_split_tensors(device_tensors)
                unpacked_tensors = packer.undo_maybe_compact_tensors(unpacked_tensors)
                unpacked_tensors = packer.undo_maybe_concat_tensors(unpacked_tensors)
                return unpacked_tensors

            with tf.device(device_tensors[0].device):
                if xla_compile:
                    unpacked_device_tensor = tf.function(unpack_single_device_tensors, jit_compile=True)()
                else:
                    unpacked_device_tensor = unpack_single_device_tensors()

            all_device_unpacked_tensors.append(unpacked_device_tensor)

        if defer_tensors:
            all_device_unpacked_tensors = _add_put_op_control_deps(
                all_device_unpacked_tensors, num_splits, all_device_put_ops)

        return all_device_unpacked_tensors, all_device_warmup_ops

    @abc.abstractmethod
    def _do_batch_all_reduce(self, all_device_tensors):
        """Performs a batch all-reduce.

        Unlike `self.batch_all_reduce`, this does not do any preprocessing of the
        tensors.

        Args:
          all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
            is a tensor where `i` is the device index and `j` is the tensor index.
        Returns:
          reduced_all_device_tensors: A list in the same form as
            `all_device_tensors`, except each tensor has been reduced.
        """
        pass


class CopyToDeviceAlgorithm(BatchAllReduceAlgorithm):
    """An algorithm that copies tensors to be reduced to a specific device."""

    def __init__(self, devices_to_reduce_on, use_mean=False):
        self._devices = devices_to_reduce_on
        self._use_mean = use_mean

    def _do_batch_all_reduce(self, all_device_tensors):
        reduced_tensors = []
        for i, tensors_across_devices in enumerate(zip(*all_device_tensors)):
            with tf.device(self._devices[i % len(self._devices)]):
                reduced_tensor = _all_reduce_using_copy(tensors_across_devices,
                                                        self._use_mean)
                reduced_tensors.append(reduced_tensor)
        return [reduced_tensors] * len(all_device_tensors)


class HierarchicalCopyAlgorithm(BatchAllReduceAlgorithm):
    """An algorithm that uses hierarchical copies. This is only optimized for
    eight devices connected in NetworkTopology.DGX1 or NetworkTopology.GCP_V100
    topology.
    """

    def __init__(self, network_topology):
        """Initializer for HierarchicalCopyAlgorithm.

        Args:
          network_topology: An instance of Enum class constants.NetworkTopology.
        """
        self._network_topology = network_topology

    def _do_batch_all_reduce(self, all_device_tensors):
        avail_devices = [device_tensors[0].device
                         for device_tensors in all_device_tensors]
        reduced_tensors = []
        num_devices = len(avail_devices)
        group_size = num_devices // 2
        for i, tensors_across_devices in enumerate(zip(*all_device_tensors)):
            group_0_main_device, group_1_main_device = self.__get_main_devices(
                i, num_devices)
            if group_0_main_device < group_size:
                group_0_begin = 0
                group_1_begin = group_size
            else:
                group_0_begin = group_size
                group_1_begin = 0

            # Reduce the first group.
            group_0_tensors = tensors_across_devices[group_0_begin:
                                                     group_0_begin + group_size]
            with tf.device(avail_devices[group_0_main_device]):
                group_0_reduced_tensor = _all_reduce_using_copy(group_0_tensors, False)

            # Reduce the second group.
            group_1_tensors = tensors_across_devices[group_1_begin:
                                                     group_1_begin + group_size]
            with tf.device(avail_devices[group_1_main_device]):
                group_1_reduced_tensor = _all_reduce_using_copy(group_1_tensors, False)

            # Reduce between the groups.
            with tf.device(avail_devices[group_0_main_device]):
                total_reduced_tensor = _all_reduce_using_copy(
                    [group_0_reduced_tensor, group_1_reduced_tensor], False)

            # Broadcast the result back into the root of each group.
            with tf.device(avail_devices[group_0_main_device]):
                group_0_reduced_tensor_bcast = tf.identity(total_reduced_tensor)
            with tf.device(avail_devices[group_1_main_device]):
                group_1_reduced_tensor_bcast = tf.identity(total_reduced_tensor)

            reduced_tensors_bcast = []
            for j in range(len(tensors_across_devices)):
                with tf.device(avail_devices[j]):
                    if (group_0_main_device < group_size) == (j < group_size):
                        src_device_tensor = group_0_reduced_tensor_bcast
                    else:
                        src_device_tensor = group_1_reduced_tensor_bcast
                    reduced_tensors_bcast.append(tf.identity(src_device_tensor))

            reduced_tensors.append(reduced_tensors_bcast)

        reduced_tensors = list(zip(*reduced_tensors))
        return reduced_tensors

    def __get_main_devices(self, tensor_index, num_devices):
        """Returns the pair of main devices to use for initial reduction.

        Args:
          tensor_index: Index of the current tensor in the list of tensors to copy.
          num_devices: Total number of devices.

        Returns:
          A tuple containing pair of main device indices for the initial
          reduction. Then, the first element of the tuple should be used for the
          final reduction.

        Raises:
          ValueError: Invalid input arguments.
        """
        if self._network_topology == constants.NetworkTopology.DGX1:
            return tensor_index % num_devices, (tensor_index +
                                                (num_devices // 2)) % num_devices
        elif self._network_topology == constants.NetworkTopology.GCP_V100:
            if num_devices != 8:
                raise ValueError('HierarchicalCopy only supports eight devices in %s.' %
                                 self._network_topology)
            main_device_pairs = [(0, 5), (2, 7), (5, 0), (7, 2)]
            return main_device_pairs[tensor_index % len(main_device_pairs)]
        else:
            raise ValueError(
                'HierarchicalCopy is not supported for %s network topology.' %
                self._network_topology)


class AllReduceSpecAlgorithm(BatchAllReduceAlgorithm):
    """An algorithm that uses an all reduce spec."""

    def __init__(self, all_reduce_spec, gpu_indices, agg_small_grads_max_bytes,
                 agg_small_grads_max_group):
        spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
        if len(spec) != 1:
            raise ValueError(
                'Replicated mode does not support hybrid all-reduce strategies')
        self._all_reduce_spec = spec[0]
        self._gpu_indices = gpu_indices
        self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
        self._agg_small_grads_max_group = agg_small_grads_max_group

    def _do_batch_all_reduce(self, all_device_tensors):
        tower_grads = [[(t, None) for t in device_tensors]
                       for device_tensors in all_device_tensors]
        aggregated_device_grads = allreduce.sum_gradients_all_reduce(
            False,  # single_session
            ['/job:localhost'],
            tower_grads,
            1,
            self._all_reduce_spec.alg,
            self._all_reduce_spec.shards,
            self._gpu_indices,
            agg_small_grads_max_bytes=self._agg_small_grads_max_bytes,
            agg_small_grads_max_group=self._agg_small_grads_max_group)
        return [[t for t, _ in grad_vars] for grad_vars in aggregated_device_grads]


def algorithm_from_params(params):
    """Returns a BatchAllReduceAlgorithm from a Params tuple."""
    if params.all_reduce_spec:
        if params.gpu_indices:
            gpu_indices = [int(x) for x in params.gpu_indices.split(',')]
        else:
            gpu_indices = [x for x in range(params.num_gpus)]
        return AllReduceSpecAlgorithm(params.all_reduce_spec, gpu_indices,
                                      params.agg_small_grads_max_bytes,
                                      params.agg_small_grads_max_group)
    elif params.hierarchical_copy:
        return HierarchicalCopyAlgorithm(params.network_topology)
    else:
        if params.local_parameter_device == 'gpu':
            devices_to_reduce_on = ['/gpu:%d' % i for i in range(params.num_gpus)]
        else:
            devices_to_reduce_on = ['/cpu:0']
        return CopyToDeviceAlgorithm(devices_to_reduce_on)


def _apply_to_all_device_tensors(all_device_tensors, apply_func, colocate=True):
    """Applies a function to each tensor in `all_device_tensors`.

    A new list of lists of tensors is returned, where every tensor in
    `all_device_tensors` has had `apply_func` called on it. `all_device_tensors`
    is not modified.

    Args:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]` is
        a tensor where `i` is the device index and `j` is the tensor index.
      apply_func: A function taking in three arguments: tensor, device_index,
        tensor_index, and returning a modified tensor.
        `tensor` is `all_device_tensors[device_index][tensor_index]`.
      colocate: If True, apply_func will be run under context manager colocated
        with its input tensor.
    Returns:
      A list in the same form as `all_device_tensors`, except each tensor has had
      `apply_func` called on it.
    """
    new_all_device_tensors = []
    for device_index, device_tensors in enumerate(all_device_tensors):
        new_device_tensors = []
        for tensor_index, t in enumerate(device_tensors):
            if colocate:
                with tf.device(t.device):
                    new_t = apply_func(t, device_index, tensor_index)
            else:
                new_t = apply_func(t, device_index, tensor_index)
            new_device_tensors.append(new_t)
        new_all_device_tensors.append(new_device_tensors)
    return new_all_device_tensors


def _defer_tensor(tensor):
    """Defers the retrieval of a tensor by using a StagingArea."""
    tensor_stage = data_flow_ops.StagingArea([tensor.dtype], [tensor.shape])
    put_op = tensor_stage.put([tensor])
    warmup_op = tensor_stage.put([tf.zeros(tensor.shape, dtype=tensor.dtype)])

    (tensor,) = tensor_stage.get()
    return tensor, put_op, warmup_op


def defer_single_device_tensors(device_tensors):
    """Defer tensors (gradients in this case) from a single device."""
    put_ops = []
    warmup_ops = []
    deferred_tensors = []

    for tensor in device_tensors:
        deferred_tensor, put_op, warmup_op = _defer_tensor(tensor)
        deferred_tensors.append(deferred_tensor)
        put_ops.append(put_op)
        warmup_ops.append(warmup_op)

    return deferred_tensors, put_ops, warmup_ops


def _add_put_op_control_deps(all_device_tensors, num_splits, put_ops):
    """Add control dependencies from `put_ops` to `all_device_tensors`."""
    def apply_func(tensor, device_index, tensor_index):
        if num_splits == 0:
            deps = [put_ops[device_index][tensor_index]]
        else:
            deps = put_ops[device_index]
        assert len(deps) == 1
        with tf.control_dependencies(deps):
            return tf.identity(tensor, name='control_dependency')
    return _apply_to_all_device_tensors(all_device_tensors, apply_func)


class _TensorPacker(object):
    """Packs and unpacks tensors into groups."""

    def __init__(self, num_splits, compact):
        """Initializes the _TensorPacker."""
        self._num_splits = num_splits
        self._compact = compact
        self._before_compact_dtypes = []

    def maybe_concat_tensors(self, device_tensors):
        """Concatenate tensors into a single tensor."""
        if not self._num_splits:
            return device_tensors

        flat_tensors = [tf.reshape(t, [-1]) for t in device_tensors]
        self._orig_shapes = [t.shape for t in device_tensors]
        self._orig_sizes = [s.num_elements() for s in self._orig_shapes]
        assert None not in self._orig_sizes
        concatenated_grad = tf.concat(flat_tensors, 0)
        return [concatenated_grad]

    def maybe_split_tensors(self, concatenated_tensor):
        """Split concatenated tensor into `num_splits` pieces."""
        if not self._num_splits:
            return concatenated_tensor

        if len(concatenated_tensor) != 1:
            raise RuntimeError('tensors must be concatenated via '
                               'maybe_concat_tensors() before splitting')

        concatenated_tensor = concatenated_tensor[0]
        total_tensor_size = concatenated_tensor.shape.num_elements()
        split_size = total_tensor_size // self._num_splits
        split_size_last = total_tensor_size - split_size * (self._num_splits - 1)
        split_sizes = [split_size] * (self._num_splits - 1) + [split_size_last]
        tensor_packs = tf.split(concatenated_tensor, split_sizes)
        return tensor_packs

    def undo_maybe_split_tensors(self, tensor_packs):
        """Undo maybe_split_tensors()."""
        if not self._num_splits:
            return tensor_packs

        return [tf.concat(tensor_packs, 0)]

    def undo_maybe_concat_tensors(self, concatenated_tensor):
        """Undo maybe_concat_tensors()."""
        if not self._num_splits:
            return concatenated_tensor

        if len(concatenated_tensor) != 1:
            raise RuntimeError(
                'undo_maybe_split_tensors() must be called before '
                'undo_maybe_concat_tensors when num_splits is greater than 1')
        concatenated_tensor = concatenated_tensor[0]

        tensors_with_sizes = tf.split(concatenated_tensor,
                                      self._orig_sizes)
        tensors_with_shapes = [
            tf.reshape(grad, shape) for grad, shape in zip(
                tensors_with_sizes, self._orig_shapes)
        ]
        return tensors_with_shapes

    def maybe_compact_tensors(self, device_tensors):
        """Cast tensors to fp16 and store their original types."""
        if not self._compact:
            return device_tensors

        if self._before_compact_dtypes:
            raise RuntimeError('maybe_compact_tensors can only be called once.')

        self._before_compact_dtypes = [t.dtype for t in device_tensors]
        compact_tensors = [tf.cast(t, tf.float16) for t in device_tensors]

        return compact_tensors

    def undo_maybe_compact_tensors(self, compact_tensors):
        """Undo maybe_compact_tensors()."""
        if not self._compact:
            return compact_tensors

        if not self._before_compact_dtypes:
            raise RuntimeError('maybe_compact_tensors() must be called before '
                               'undo_maybe_compact_tensors()')

        device_tensors = [
            tf.cast(t, dtype)
            for t, dtype in zip(compact_tensors, self._before_compact_dtypes)
        ]
        return device_tensors