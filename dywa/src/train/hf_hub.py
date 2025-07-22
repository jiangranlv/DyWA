#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from util.config import ConfigBase


@dataclass
class GroupConfig(ConfigBase):
    # one of imm,mc,jh,xai,vessl
    # set by +platform=
    machine: str = ''
    # one of "push" "hand" "arm"
    # and any other extra identifiers
    # set by +env=
    env_name: str = ''
    model_name: str = ''
    tag: str = ''


@dataclass
class HfConfig(ConfigBase):
    use_hfhub: bool = True
    hf_repo_id: Optional[str] = None  # 'corn/corn-/wrench'


def upload_ckpt(repo_id: str,
                ckpt_file: str,
                name: Optional[str] = None):
    ckpt_file = Path(ckpt_file)

    api = HfApi()

    # 1. Create repo.
    url = api.create_repo(repo_id,
                          repo_type=None,  # =="model"
                          exist_ok=True)

    # 2. [Optional] auto-configure name
    if name is None:
        name = ckpt_file.name

    # 3. Upload file.
    api.upload_file(
        path_or_fileobj=str(ckpt_file),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type='model'
    )


def download_ckpt(repo_id: str, name: str) -> str:
    """ Download checkpoint from huggingface model hub. """
    ckpt_file: str = hf_hub_download(repo_id, name)
    return ckpt_file
