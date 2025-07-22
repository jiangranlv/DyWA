#!/usr/bin/env python3

import os
from pathlib import Path
from typing import (
    Callable, List, Type,
    TypeVar, Union
)
from dataclasses import (dataclass, is_dataclass)
import importlib.resources

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from hydra.types import ConvertMode

from functools import wraps
import inspect
from icecream import ic


D = TypeVar("D")


def default_cfg_path() -> str:
    # [1] highest priority, read from env
    pkm_cfg_path: str = os.getenv('PKM_CFG_PATH', None)
    if pkm_cfg_path is not None:
        return pkm_cfg_path

    # [2] fallback to data.cfg
    with importlib.resources.path('data', 'cfg') as rp:
        return str(rp)

    # [3] fallback to PWD
    return '.'


def hydra_cli(
        cls: Union[Type[D], D] = None,
        argv: List[str] = None,
        config_path: str = default_cfg_path(),
        config_name: str = 'hydra'):
    """
    Decorator for automatically adding parsed args from cli
    to entry point, based on hydra backend.
    """
    if argv is not None:
        raise ValueError('I realized that argv is actually ignored in hydra. Do not use!')

    # Logic for handling different forms of
    # the decorator ...
    main: Callable = None
    need_cls: bool = True

    if is_dataclass(cls):
        # @with_cli(cls=Config ...)
        need_cls = False
    elif callable(cls):
        # @with_cli
        main = cls
    else:
        # Here, probably invoked as
        # @with_cli()
        pass

    def decorator(main: Callable[[D], None]):
        # NOTE:
        # if `cls` is None,
        # try to infer them from `main` signature.
        cfg_cls = cls
        if need_cls:
            sig = inspect.signature(main)
            if len(sig.parameters) == 1:
                key = next(iter(sig.parameters))
                cfg_cls = sig.parameters[key].annotation
            else:
                raise ValueError(
                    '#arg != 1 in main {}: Cannot infer param type.'
                    .format(sig))

        # NOTE: using @wraps to forward main() documentation.
        # NOTE: hydra only works with relative path...
        rel_path: str = os.path.relpath(config_path,
                                        start=Path(__file__).parent.resolve())

        @wraps(main)
        @hydra.main(
            config_path=rel_path,
            config_name=config_name,
            version_base=None)
        def wrapper(cli_cfg: DictConfig):
            # Load defaults from `cls`, either
            # default-constructed or from passed instance.
            default_cfg = OmegaConf.structured(cfg_cls)
            # ic(default_cfg)
            # Load updates from the CLI, where hydra handles
            # the config files and command-line arguments.

            # cli_cfg=OmegaConf.merge(cli_cfg,default_cfg,cli_cfg)
            # ic(cli_cfg)
            # cli_cfg = cli_cfg.structured(True)
            # OmegaConf.set_struct(cli_cfg,True)
            # ic(cli_cfg)
            cli_cfg = hydra.utils.instantiate(cli_cfg,
                                              _convert_=ConvertMode.PARTIAL)
            ic(cli_cfg)

            # Merge and return
            cfg = OmegaConf.to_object(
                OmegaConf.structured(
                    OmegaConf.merge(default_cfg, cli_cfg)
                )
            )
            return main(cfg)
        return wrapper

    if main is not None:
        return decorator(main)
    else:
        return decorator


def main():
    @dataclass
    class Config:
        a: int = 2
        b: int = 3

    import tempfile
    from textwrap import dedent

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(F'{tmpdir}/hydra.yaml', 'w') as fp:
            fp.write(dedent("""
            defaults:
              - override hydra/hydra_logging: disabled
              - override hydra/job_logging: disabled
            hydra:
              output_subdir: null
              run:
                dir: .
            # a: 3
            """))

        @hydra_cli(Config(a=5), config_path=tmpdir, config_name='hydra')
        def _main(cfg: Config):
            """ DO STUFF """
            print('cfg', cfg)
            print(type(cfg))
            pass
        _main()


if __name__ == '__main__':
    main()
