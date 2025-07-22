#!/usr/bin/env python3

from os import PathLike
from typing import Union, Any, Dict, Callable
import torch as th
from pathlib import Path

from util.path import ensure_directory
from train.hf_hub import hf_hub_download
from typing import Optional, List


def step_from_ckpt(ckpt_file: str) -> float:
    """
    Brittle string-parse operation to get the `global_step`
    parameter from the name of the checkpoint.
    No one should rely on this function.
    """
    ckpt_file = str(ckpt_file)
    if 'last' in ckpt_file:
        return float('inf')
    try:
        return int(Path(ckpt_file).stem.split('-')[1])
    except ValueError:  # numerical conversion failed
        return -1


def save_ckpt(modules: Dict[str, Union[dict, th.nn.Module]],
              ckpt_file: str):
    """ Save checkpoint. """
    ckpt_file = Path(ckpt_file)
    ensure_directory(ckpt_file.parent)
    save_dict: Dict[str, Any] = {}

    for k, v in modules.items():
        if isinstance(v, th.nn.DataParallel):
            v = v.module
        if isinstance(v, th.nn.Module) or hasattr(v, 'state_dict'):
            save_dict[k] = v.state_dict()
        elif isinstance(v, dict):
            save_dict[k] = v
    th.save(save_dict, str(ckpt_file))


def load_ckpt(modules: Dict[str, Union[dict, th.nn.Module]],
              ckpt_file: str, strict: bool = True, exclude_keys:Optional[List[str]] = None):
    """ Load checkpoint. """
    if isinstance(ckpt_file, dict):
        save_dict = ckpt_file
    else:
        ckpt_file = Path(ckpt_file)
        save_dict = th.load(str(ckpt_file),
                            map_location='cpu')
    
    if exclude_keys:
        for k, m in modules.items():
            for key in exclude_keys:
                save_dict[k].pop(key, None)

    for k, m in modules.items():
        if isinstance(m, th.nn.DataParallel):
            m = m.module
        try:
            if isinstance(m, th.nn.Module):
                m.load_state_dict(save_dict[k],
                                  strict=strict)
            else:
                m.load_state_dict(save_dict[k])
        except KeyError as e:
            if strict:
                raise
            else:
                print(F'Encountered error during `load_ckpt`: {e}')


def last_ckpt(root: Union[str, PathLike, Path],
              pattern: str = '*.ckpt',
              key: Callable[[Path], Any] = None):

    # By default, sort by file modification time.
    if key is None:
        lambda f: f.stat().st_mtime

    path = Path(root)
    if path.is_file():
        return path

    try:
        last_ckpt = max(path.rglob(pattern), key=key)
    except ValueError:
        # Fallback to huggingface
        repo_id, ckpt_name = str(root).split(':', maxsplit=1)
        last_ckpt = hf_hub_download(repo_id, ckpt_name)

    return last_ckpt
