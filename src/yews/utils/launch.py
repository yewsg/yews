import logging
import socket
from typing import Callable, Optional

import torch
from torch import distributed as dist

from . import comm

# binding to port 0 will cause the OS to find an available port
# there is still a chance the port could be taken by other processes
def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# distributed worker
def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
) -> None:

    # env checks
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    if not torch.cuda.is_available():
        raise ValueError("Cuda is not available. Please check your installation.")
    if num_gpus_per_machine > torch.cuda.device_count():
        raise ValueError("Cannot use more GPU devices than available.")
    torch.cuda.set_device(local_rank)

    # init process group
    try:
        dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank)
        # main_func(*args)
    except Exception as error:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: %s", dist_url)
        raise error
    comm.synchronize()

    # setup local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        process_group = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = process_group

    main_func(*args)


# launch multi gpu or distributed training
# this function must be called on all machines involved in the training
# it will spawn child processes (defined by num_gpus_per_machine) on each machine
def launch(
    main_func: Callable,
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: Optional[str] = None,
    start_method: str = "spawn",
    args: tuple = (),
) -> None:
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url is None:
            if num_machines > 1:
                raise ValueError("Please manually define dist_url for multi-machine training.")
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        torch.multiprocessing.start_processes(
            _distributed_worker,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
            nprocs=num_gpus_per_machine,
            daemon=False,
            start_method=start_method,
        )
    else:
        main_func(*args)
