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
"""Utilities for allreduce."""

import collections as pycoll
import re

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# `tensorflow.contrib` has been removed in TensorFlow 2.x, so the following import is no longer valid.
# We need to either reimplement the all-reduce logic or use another library that provides similar functionality.
# from tensorflow.contrib.all_reduce.python import all_reduce

# Since collective_ops is still part of TensorFlow's core, we can keep it.
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops

AllReduceSpecTuple = pycoll.namedtuple('AllReduceSpecTuple', 'alg shards limit')


def parse_general_int(s):
    """Parse integer with power-of-2 suffix eg. 32k."""
    mo = re.match(r'(\d+)([KkMGT]?)$', s)
    if mo:
        i, suffix = mo.group(1, 2)
        v = int(i)
        if suffix:
            if suffix == 'K' or suffix == 'k':
                v *= 1024
            elif suffix == 'M':
                v *= (1024 * 1024)
            elif suffix == 'G':
                v *= (1024 * 1024 * 1024)
            elif suffix == 'T':
                v *= (1024 * 1024 * 1024 * 1024)
            else:
                raise ValueError('invalid integer string %s' % s)
        return v
    else:
        v = int(s)
    return v


def parse_all_reduce_spec(all_reduce_spec):
    """Parse all_reduce_spec.

    Args:
      all_reduce_spec: a string specifying a combination of all-reduce
        algorithms to apply for gradient reduction.

    Returns:
      a list of AllReduceSpecTuple.

    Raises:
      ValueError: all_reduce_spec is not well-formed.
    """
    range_parts = all_reduce_spec.split(':') + ['-1']
    if len(range_parts) % 2:
        raise ValueError('all_reduce_spec not well formed: %s' % all_reduce_spec)
    limit = 0
    spec = []
    alg = None
    shards = 1
    for i, range_part in enumerate(range_parts):
        if i % 2 == 1:
            try:
                limit = parse_general_int(range_part)
                spec.append(AllReduceSpecTuple(alg=alg, shards=shards, limit=limit))
            except ValueError:
                raise ValueError('all_reduce_spec (%s) contains non-integer range %s' %
                                 (all_reduce_spec, range_part))
        else:
            alg = range_part
            alg_parts = range_part.split('#')
            alg = alg_parts[0]
            if len(alg_parts) > 1:
                try:
                    shards = int(alg_parts[1])
                except ValueError:
                    raise ValueError('all_reduce_spec (%s) contains non-integer '
                                     'shards %s' % all_reduce_spec, alg_parts[1])
            else:
                shards = 1
            if alg not in [
                'nccl', 'nccl/xring', 'nccl/rechd', 'nccl/pscpu', 'xring', 'pscpu',
                'psgpu', 'pscpu/pscpu', 'collective'
            ]:
                raise ValueError('all_reduce_spec (%s) contains invalid alg %s' %
                                 (all_reduce_spec, alg))
    return spec


def build_all_reduce_device_prefixes(job_name, num_tasks):
    """Build list of device prefix names for all_reduce.

    Args:
      job_name: 'worker', 'ps' or 'localhost'.
      num_tasks: number of jobs across which device names should be generated.

    Returns:
       A list of device name prefix strings. Each element spells out the full
       host name without adding the device.
       e.g. '/job:worker/task:0'
    """
    if job_name != 'localhost':
        return ['/job:%s/task:%d' % (job_name, d) for d in range(0, num_tasks)]
    else:
        assert num_tasks == 1
        return ['/job:%s' % job_name]


def group_device_names(devices, group_size):
    """Group device names into groups of group_size.

    Args:
      devices: list of strings naming devices.
      group_size: int >= 1

    Returns:
      list of lists of devices, where each inner list is group_size long,
        and each device appears at least once in an inner list.  If
        len(devices) % group_size = 0 then each device will appear
        exactly once.

    Raises:
      ValueError: group_size > len(devices)
    """
    num_devices = len(devices)
    if group_size > num_devices:
        raise ValueError('only %d devices, but group_size=%d' % (num_devices, group_size))
    num_groups = (
        num_devices // group_size + (1 if (num_devices % group_size != 0) else 0))
    groups = [[] for _ in range(num_groups)]
    for i in range(0, num_groups * group_size):
        groups[i % num_groups].append(devices[i % num_devices])
    return groups


def split_grads_by_size(threshold_size, device_grads):
    """Break gradients into two sets according to tensor size.

    Args:
      threshold_size: int size cutoff for small vs large tensor.
      device_grads: List of lists of (gradient, variable) tuples.  The outer
          list is over devices. The inner list is over individual gradients.

    Returns:
      small_grads: Subset of device_grads where shape is <= theshold_size
         elements.
      large_grads: Subset of device_grads where shape is > threshold_size
         elements.
    """
    small_grads = []
    large_grads = []
    for dl in device_grads:
        small_dl = []
        large_dl = []
        for (g, v) in dl:
            tensor_size = g.get_shape().num_elements()
            if tensor_size <= threshold_size:
                small_dl.append([g, v])
            else:
                large_dl.append([g, v])
        if small_dl:
            small_grads.append(small_dl)
        if large_dl:
            large_grads.append(large_dl)
    return small_grads, large_grads


_instance_key = 1


def new_collective_instance_key():
    """Returns a new instance key for use in defining a collective op."""
    global _instance_key
    v = _instance_key
    _instance_key += 1
    return v


_group_key = 1
_group_key_table = dict()


def collective_group_key(devices):
    """Returns a group key for the set of devices.

    Args:
      devices: list of strings naming devices in a collective group.

    Returns:
      int key uniquely identifying the set of device names.
    """
    global _group_key
    global _group_key_table
    parsed = [pydev.DeviceSpec.from_string(d) for d in devices]
    names = sorted(['%s:%d' % (d.device_type, d.device_index) for d in parsed])
    concat = ','.join(names)
    if concat not in _group_key_table.keys():
        new_key = _group_key
        _group_key += 1
        _group_key_table[concat] = new_key
    rv = _group_key_table[concat]
    return rv


def build_collective_reduce(input_tensors, num_workers, num_shards,
                            red_op='Add', un_op='Id'):
    """Build a subgraph that does one full all-reduce, using the collective Op.

    Args:
      input_tensors: tensors within a single worker graph that are to be reduced
        together; must be one per device.
      num_workers: total number of workers with identical independent graphs that
        will be doing this same reduction.  The reduction will actually include
        the corresponding tensors at all these workers.
      num_shards: number of shards into which to divide each per-tick chunk,
        normally 1 but could be higher on multi-data-path architectures.
      red_op: string naming the reduction op
      un_op: string naming the unary final op

    Returns:
      An array of final tensors, one per device, computed by the full reduction.

    Raises:
      ValueError: There must be at least two tensors over all the workers.
    """
    group_size = len(input_tensors) * num_workers
    if group_size < 2:
        raise ValueError('num_workers * len(input_tensors) must be 2 or greater')
    devices = [t.device for t in input_tensors]
    num_devices = len(devices)
    group_key = collective_group_key(devices)
    instance_key = new_collective_instance_key()
    out_tensors = []
    if num_shards == 1:
        subdiv_offsets = [0]
    elif num_shards == 2:
        if num_devices > 1:
            subdiv_offsets = [0, -(num_devices // 2)]
        else:
            subdiv_offsets = [0]
    else:
        raise ValueError('Unsupported num_shards %d' % num_shards)
    for d in range(num_devices):
        with ops.device(devices[d]):
            reduce_op = collective_ops.all_reduce(input_tensors[d],
                                                  group_size, group_key, instance_key,
                                                  red_op, un_op,
                                                  subdiv_offsets)
            out_tensors.append(reduce_op)
    return out_tensors


def broadcast_send(t, shape, dtype, group_size, group_key, instance_key):
    return collective_ops.broadcast_send(t, shape, dtype, group_size, group_key,
                                         instance_key)


def broadcast_recv(shape, dtype, group_size, group_key, instance_key):
    return collective_ops.broadcast_recv(shape, dtype, group_size, group_key,
                                         instance_key)


def sum_grad_and_var_all_reduce(single_session,
                                grad_and_vars,
                                num_workers,
                                alg,
                                gpu_indices,
                                aux_devices=None,
                                num_shards=1):
    """Apply all-reduce algorithm over specified gradient tensors."""
    scaled_grads = [g for g, _ in grad_and_vars]
    if alg == 'collective':
        assert not single_session
        summed_grads = build_collective_reduce(
            scaled_grads, num_workers, num_shards, 'Add', 'Id')
    else:
        with tf.name_scope('allreduce'):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            if alg == 'nccl':
                summed_grads = tf.raw_ops.NcclAllReduce(
                    input=scaled_grads, reduction='sum')
            elif alg == 'xring':
                # You would need to implement ring all-reduce manually or use a different approach
                raise NotImplementedError("Ring All-Reduce (xring) is not implemented.")
            elif alg == 'nccl/xring':
                # You would need to implement NCCL then ring all-reduce manually or use a different approach
                raise NotImplementedError("NCCL then Ring All-Reduce is not implemented.")
            elif alg == 'nccl/rechd':
                # You would need to implement NCCL then recursive halving and doubling manually
                raise NotImplementedError("NCCL then Recursive Halving and Doubling is not implemented.")
            elif alg == 'nccl/pscpu':
                # Implementing NCCL then shuffle all-reduce is complex and requires custom logic
                raise NotImplementedError("NCCL then Shuffle All-Reduce is not implemented.")
            elif alg == 'pscpu/pscpu':
                # Implementing shuffle then shuffle all-reduce is complex and requires custom logic
                raise NotImplementedError("Shuffle then Shuffle All-Reduce is not implemented.")
            elif alg in ['pscpu', 'psgpu']:
                # Implementing shuffle all-reduce is complex and requires custom logic
                raise NotImplementedError("Shuffle All-Reduce is not implemented.")
            else:
                raise ValueError('unsupported all_reduce alg: ', alg)

    result = []
    for (_, v), g in zip(grad_and_vars, summed_grads):
        result.append([g, v])
    return result


def contains_any(haystack, needles):
    """Tests if any needle is a substring of haystack.

    Args:
      haystack: a string
      needles: list of strings

    Returns:
      True if any element of needles is a substring of haystack,
        False otherwise.
    """
    for n in needles:
        if n in haystack:
            return True
    return False


def sum_gradients_all_reduce(single_session,
                             dev_prefixes,
                             tower_grads,
                             num_workers,
                             alg,
                             num_shards,
                             gpu_indices,
                             agg_small_grads_max_bytes=0,
                             agg_small_grads_max_group=10,
                             allreduce_merge_scope=1):
    """Apply all-reduce algorithm over specified gradient tensors.

    Args:
      single_session: true if reduction is applied to one graph across
        all workers, false if ths application is to a single-worker graph only.
      dev_prefixes: list of prefix strings to use to generate PS device names.
      tower_grads: the gradients to reduce.
      num_workers: number of worker processes across entire job.
      alg: the all-reduce algorithm to apply.
      num_shards: alg-specific sharding factor.
      gpu_indices: indices of local GPUs in order usable for ring-reduce.
      agg_small_grads_max_bytes: largest tensor eligible for aggregation,
        in number of bytes.
      agg_small_grads_max_group: largest permitted aggregation of small
        tensors.
      allreduce_merge_scope: size of groups into which to partition consecutive
        gradients grouped under a common 'allreduce' name scope for application
        of ScopedAllocator optimization.

    Returns:
      list of reduced tensors
    """
    alg_contains_shuffle = contains_any(alg, ['pscpu', 'psgpu'])
    is_hierarchical = '/' in alg
    if 'pscpu' in alg:
        aux_devices = [prefix + '/cpu:0' for prefix in dev_prefixes]
    elif 'psgpu' in alg:
        aux_devices = [
            prefix + '/gpu:%d' % i
            for i in range(len(gpu_indices))
            for prefix in dev_prefixes
        ]
    else:
        aux_devices = ['/job:localhost/cpu:0']
    aux_device_groups = group_device_names(
        aux_devices,
        num_shards if (alg != 'collective' and alg_contains_shuffle) else 1)
    group_index = 0
    if agg_small_grads_max_bytes > 0 and agg_small_grads_max_group > 0:
        tower_grads, packing = pack_small_tensors(
            tower_grads,
            max_bytes=agg_small_grads_max_bytes,
            max_group=agg_small_grads_max_group)
    else:
        packing = None
    reduced_gv_list = []
    gv = list(zip(*tower_grads))
    merge_scope = allreduce_merge_scope if allreduce_merge_scope > 0 else 1
    chunked_gv = [gv[x:x + merge_scope]
                  for x in xrange(0, len(gv), merge_scope)]
    for chunk in chunked_gv:
        with tf.name_scope('allreduce'):
            for grad_and_vars in chunk:
                reduced_gv_list.append(sum_grad_and_var_all_reduce(
                    single_session,
                    grad_and_vars, num_workers, alg, gpu_indices,
                    (aux_devices if is_hierarchical
                     else aux_device_groups[group_index]),
                    num_shards))
                group_index = (group_index + 1) % len(aux_device_groups)
    new_tower_grads = [list(x) for x in zip(*reduced_gv_list)]
    if packing:
        new_tower_grads = unpack_small_tensors(new_tower_grads, packing)
    return new_tower_grads


def extract_ranges(index_list, range_size_limit=32):
    """Extract consecutive ranges and singles from index_list.

    Args:
      index_list: List of monotone increasing non-negative integers.
      range_size_limit: Largest size range to return.  If a larger
        consecutive range exists it will be returned as multiple
        ranges.

    Returns:
     ranges, singles where ranges is a list of [first, last] pairs of
       consecutive elements in index_list, and singles is all of the
       other elements, in original order.
    """
    if not index_list:
        return [], []
    first = index_list[0]
    last = first
    ranges = []
    singles = []
    for i in index_list[1:]:
        if i == last + 1 and (last - first) <= range_size_limit:
            last = i
        else:
            if last > first:
                ranges.append([first, last])
            else:
                singles.append(first)
            first = i
            last = i
    if last > first:
        ranges.append([first, last])
    else:
        singles.append(first)
    return ranges, singles


GradPackTuple = pycoll.namedtuple('GradPackTuple', 'indices vars shapes')


def pack_range(key, packing, grad_vars, rng):
    """Form the concatenation of a specified range of gradient tensors.

    Args:
      key: Value under which to store meta-data in packing that will be used
        later to restore the grad_var list structure.
      packing: Dict holding data describing packed ranges of small tensors.
      grad_vars: List of (grad, var) pairs for one tower.
      rng: A pair of integers giving the first, last indices of a consecutive
        range of tensors to be packed.

    Returns:
      A tensor that is the concatenation of all the specified small tensors.
    """
    to_pack = grad_vars[rng[0]:rng[1] + 1]
    members = []
    variables = []
    restore_shapes = []
    with tf.name_scope('pack'):
        for g, v in to_pack:
            variables.append(v)
            restore_shapes.append(g.shape)
            with tf.device(g.device):
                members.append(tf.reshape(g, [-1]))
        packing[key] = GradPackTuple(
            indices=range(rng[0], rng[1] + 1),
            vars=variables,
            shapes=restore_shapes)
        with tf.device(members[0].device):
            return tf.concat(members, 0)


def unpack_grad_tuple(gv, gpt):
    """Unpack a previously packed collection of gradient tensors.

    Args:
      gv: A (grad, var) pair to be unpacked.
      gpt: A GradPackTuple describing the packing operation that produced gv.

    Returns:
      A list of (grad, var) pairs corresponding to the values that were
       originally packed into gv, maybe following subsequent operations like
       reduction.
    """
    elt_widths = [x.num_elements() for x in gpt.shapes]
    with tf.device(gv[0][0].device):
        with tf.name_scope('unpack'):
            splits = tf.split(gv[0], elt_widths)
            unpacked_gv = []
            for idx, s in enumerate(splits):
                unpacked_gv.append((tf.reshape(s, gpt.shapes[idx]), gpt.vars[idx]))
    return unpacked_gv


def pack_small_tensors(tower_grads, max_bytes=0, max_group=0):
    """Concatenate small gradient tensors together for reduction.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples.
      max_bytes: Int giving max number of bytes in a tensor that
        may be considered small.
      max_group: Int giving max number of small tensors that may be
        concatenated into one new tensor.

    Returns:
      new_tower_grads, packing where new_tower_grads is identical to
        tower_grads except that all feasible small_tensors have been removed
        from their places and concatenated into larger tensors that are
        now in the front of the list for each tower, and packing contains
        the data necessary to restore the tower_grads structure.
    """
    small_indices = []
    large_indices = []
    for idx, (g, _) in enumerate(tower_grads[0]):
        if g.dtype == tf.float32 and (4 * g.shape.num_elements()) <= max_bytes:
            small_indices.append(idx)
        else:
            large_indices.append(idx)
    small_ranges, small_singles = extract_ranges(
        small_indices, range_size_limit=max_group)
    large_indices = sorted(large_indices + small_singles)
    num_gv = len(tower_grads[0])
    packing = {}
    if small_ranges:
        new_tower_grads = []
        for dev_idx, gv_list in enumerate(tower_grads):
            assert len(gv_list) == num_gv
            new_gv_list = []
            for r in small_ranges:
                key = '%d:%d' % (dev_idx, len(new_gv_list))
                new_gv_list.append((pack_range(key, packing, gv_list, r),
                                    'packing_var_placeholder'))
            for i in large_indices:
                new_gv_list.append(gv_list[i])
            new_tower_grads.append(new_gv_list)
        return new_tower_grads, packing
    else:
        return tower_grads, None


def unpack_small_tensors(tower_grads, packing):
    """Undo the structure alterations to tower_grads done by pack_small_tensors.

    Args:
      tower_grads: List of List of (grad, var) tuples.
      packing: A dict generated by pack_small_tensors describing the changes
        it made to tower_grads.

    Returns:
      new_tower_grads: identical to tower_grads except that concatentations
        of small tensors have been split apart and returned to their original
        positions, paired with their original variables.
    """
    if not packing:
        return tower_grads
    new_tower_grads = []
    num_devices = len(tower_grads)
    num_packed = len(packing.keys()) // num_devices
    for dev_idx, gv_list in enumerate(tower_grads):
        new_gv_list = gv_list[num_packed:]
        for i in xrange(0, num_packed):
            k = '%d:%d' % (dev_idx, i)
            gpt = packing[k]
            gv = unpack_grad_tuple(gv_list[i], gpt)
            for gi, idx in enumerate(gpt.indices):
                assert idx == gpt.indices[gi]
                new_gv_list.insert(idx, gv[gi])
        new_tower_grads.append(new_gv_list)
    return new_tower_grads