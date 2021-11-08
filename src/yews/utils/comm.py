"""Utilities to facilitate distributed training.

Adapted from Detectron2 utilities:
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py

"""
import functools
import logging
import pickle
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import distributed as dist

# a torch process group which only includes processes that on the same machine
# as the current process; this variable is set when processes are started by
# `launch()`.
_LOCAL_PROCESS_GROUP = None

# check if current process is in distributed mode
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


# get the total number of processes in distributed mode
def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


# get rank of the current process
def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


# rank of the current process within the local process group
def get_local_rank() -> int:
    if not is_distributed():
        return 0
    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError("_LOCAL_PROCESS_GROUP must be preset set in _distributed_worker() method.")
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


# the size of the per-machine process group, i.e. the number of processes per machine.
def get_local_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


# determines if the current process is the master process; the master process is responsible for logging,
# writing and loading checkpoints; in the multi GPU setting, we assign the master role to the rank 0 process
def is_main_process() -> bool:
    return get_rank() == 0


# synchronize (barrier) among all processes in distributed mode
# https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier(device_ids=[get_rank()])


# return a process group based on gloo backend, containing all the ranks; the result is cached
@functools.lru_cache()
def _get_global_gloo_group() -> object:
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


# serialize an object to a tensor
def _serialize_to_tensor(data: Any, group: object) -> torch.Tensor:
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


# pad the tensor because torch all_gather does not support gathering tensors of different shapes
# return a list[int]: size of the tensor, on each rank and padded tensor that has the max size
def _pad_to_largest_tensor(tensor: torch.Tensor, group: object) -> Tuple[list, torch.Tensor]:
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)  # pylint: disable=not-callable
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


# run all_gather on arbitrary picklable data (not necessarily tensors)
# data is any picklable object and group is a torch process group; by default, will use a group which contains all ranks on gloo backend
# returns a list of data gathered from each rank
def all_gather(data: Any, group: Optional[object] = None) -> list:
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


# run gather on arbitrary picklable data (not necessarily tensors)
# data is any picklable object, dst is a destination rank and group is a torch process group; by default, will use a group which contains all ranks on gloo backend
# returns a list of data gathered from each rank; otherwise, an empty list
def gather(data: Any, dst: int = 0, group: object = None) -> list:
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)
        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


# a random number that is the same across all workers; if workers need a shared RNG, they can use
# this shared seed to create one; all workers must call this function, otherwise it will deadlock
def shared_random_seed() -> int:
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


# reduce the values in the dictionary from all processes so that process with rank 0 has the reduced results
# input_dict inputs to be reduced; all the values must be scalar CUDA tensor
# average indicates whether to do average or sum
# returns a dict with the same keys as input_dict, after reduction
def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
