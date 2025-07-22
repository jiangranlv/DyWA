#!/usr/bin/env python3

from typing import Optional, Callable
from util.config import ConfigBase
from dataclasses import dataclass, is_dataclass
from omegaconf import OmegaConf
import functools
import wandb


@dataclass
class WandbConfig(ConfigBase):
    project: Optional[str] = ''
    sync_tensorboard: bool = True
    save_code: bool = True
    use_wandb: bool = True
    # It's technically "optional", but required if
    # `use_wandb`=True.
    name: Optional[str] = None
    group: Optional[str] = None
    entity: Optional[str] = None
    # Optional arg if you want to resume
    run_id: Optional[str] = None


def with_wandb(main: Callable):
    """ Lightweight WandB experiment wrapper. """

    @functools.wraps(main)
    def wrapper(cfg: WandbConfig, *args, **kwds):
        assert (cfg.project is not None)
        assert (isinstance(cfg.project, str))
        assert (len(cfg.project) > 0)
        # NOTE: this assert requires that
        # `name` will not be set to the
        # random default from wandb.
        assert (cfg.name is not None)
        if cfg.use_wandb:
            wandb.login(
                anonymous='allow',
                relogin=False,
                force=False)
            if hasattr(cfg, 'to_dict'):
                # simple_parsing.Serializable
                cfg_dict = cfg.to_dict()
            else:
                # OmegaConf
                if is_dataclass(cfg):
                    oc_cfg = OmegaConf.structured(cfg)
                else:
                    oc_cfg = cfg
                cfg_dict = OmegaConf.to_container(oc_cfg)

            # Remaining configs I'm not too sure about:
            # dir: Union[str, pathlib.Path, None] = None,
            # tags: Optional[Sequence] = None,
            # name: Optional[str] = None,
            # resume: Optional[Union[bool, str]] = None,
            # wandb.tensorboard.patch(root_logdir=log_dir)
            wandb.init(
                project=cfg.project,
                sync_tensorboard=cfg.sync_tensorboard,
                save_code=cfg.save_code,
                config=cfg_dict,
                group=cfg.group,
                name=cfg.name,
                entity=cfg.entity,
                id=cfg.run_id,
                resume='allow',
                # entity='corn'
            )
        try:
            main(cfg, *args, **kwds)
        finally:
            if cfg.use_wandb:
                wandb.finish()

    return wrapper
