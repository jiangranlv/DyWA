#!/usr/bin/env python3

import pkg_resources
import logging
from os import PathLike
from typing import Optional, Union
from pathlib import Path
from util.config import ConfigBase
from dataclasses import dataclass
from tempfile import TemporaryDirectory


def get_path(path: str) -> str:
    """Get resource path."""
    return pkg_resources.resource_filename('data', path)


def get_latest_file(path: Path, pattern: str = '*'):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(F'path {path} is not a directory.')
    latest_file = max(path.glob(pattern),
                      key=lambda p: p.stat().st_mtime)
    return latest_file


def ensure_directory(path: Union[str, PathLike]) -> Path:
    """Ensure that the directory structure exists."""
    path = Path(path)
    if path.is_dir():
        return path

    if path.exists():
        raise ValueError(F'path {path} exists and is not a directory.')
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise ValueError(F'Failed to create path {path}.')

    return path


def get_new_dir(root: Union[str, PathLike]) -> Path:
    """Get new runtime directory."""
    root = Path(root).expanduser()
    index = len([d for d in root.glob('run-*') if d.is_dir()])
    path = ensure_directory(root / F'run-{index:03d}')
    return path


class RunPath(object):
    """General path management over multiple experiment runs.

    NOTE: The intent of this class is mainly to avoid overwriting
    checkpoints and existing logs from a previous run -
    instead, we maintain a collision-free index based key
    for each experiment that we run and use them in a sub-folder structure.
    """

    @dataclass
    class Config(ConfigBase):
        key_format: str = 'run-{:03d}'
        root: Optional[str] = '/tmp/'  # Alternatively, ~/.cache/pkm/run/
        key: Optional[str] = None  # Empty string indicates auto increment.

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if (cfg.root is None or
                (isinstance(cfg.root, str) and cfg.root.lower() == 'none')):
            root = TemporaryDirectory()
        else:
            root = ensure_directory(
                Path(cfg.root).expanduser())
        self._root_path = root
        self.root = Path(
            root.name if isinstance(root, TemporaryDirectory) else root
        )

        # Resolve sub-directory key.
        key = cfg.key
        if key is None:
            key = self._resolve_key(self.root, self.cfg.key_format)
            logging.info(F'key={key}')

        self.dir = ensure_directory(self.root / key)
        logging.info(F'self.dir={self.dir}')

    def __del__(self):
        if isinstance(self._root_path, TemporaryDirectory):
            self._root_path.cleanup()

    @staticmethod
    def _resolve_key(root: str, key_fmt: str) -> str:
        """Get latest valid key according to `key_fmt`"""
        # Ensure `root` is a valid directory.
        root = Path(root)
        if not root.is_dir():
            raise ValueError(F'Arg root={root} is not a dir.')

        # NOTE: Loop through integers starting from 0.
        # Not necessarily efficient, but convenient.
        index = 0
        while True:
            key = key_fmt.format(index)
            if not (root / key).exists():
                break
            index += 1
        return key

    def __getattr__(self, key: str):
        """
        Convenient shorthand for fetching valid subdirectories.
        """
        return ensure_directory(self.dir / key)


def test_init():
    with TemporaryDirectory() as tmpdir:
        rp = RunPath(RunPath.Config(root=tmpdir))
        logging.debug(rp.log)
        rp2 = RunPath(RunPath.Config(root=tmpdir))
        logging.debug(rp2.log)


def test_tempdir():
    def _main():
        rp = RunPath(RunPath.Config(root=None))
        print(rp)
        assert (Path(rp._root_path.name).exists())  # should be `True`
        return rp._root_path.name
    path = _main()
    assert (not Path(path).exists())  # should be `False`


def test_parse_config():
    config_yaml_str = 'root: null'
    print(RunPath.Config.loads_yaml(config_yaml_str))


def main():
    test_parse_config()


if __name__ == '__main__':
    main()
