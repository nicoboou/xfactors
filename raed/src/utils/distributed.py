from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def set_cuda_visible_devices(gpu_ids: list[int] | None) -> None:
    if not gpu_ids:
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(cfg: dict[str, Any]) -> tuple[torch.device, int]:
    runtime_cfg = cfg.get("runtime", {})
    use_ddp = bool(runtime_cfg.get("use_ddp", False))
    if use_ddp:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = runtime_cfg.get("backend", "nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return device, local_rank

    if torch.cuda.is_available():
        return torch.device("cuda", 0), 0
    return torch.device("cpu"), 0


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def ddp_wrap(model: torch.nn.Module, device: torch.device, find_unused_parameters: bool = False) -> torch.nn.Module:
    if not is_distributed():
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=find_unused_parameters)
    return DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return float(tensor.item())


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    return {key: reduce_mean_scalar(val, device) for key, val in metrics.items()}


def state_dict_for_save(model: torch.nn.Module) -> dict[str, Any]:
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()
