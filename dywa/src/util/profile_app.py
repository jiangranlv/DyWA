#!/usr/bin/env python3

__all__ = ['profile', 'line_profile']

from contextlib import contextmanager
import cProfile
import pstats
try:
    from line_profiler import LineProfiler
except ModuleNotFoundError:
    pass
from dataclasses import dataclass
from util.config import ConfigBase
from typing import Tuple, Callable, Union
import importlib
import inspect


@contextmanager
def profile(key: str = 'cumtime', lines: int = 32,
            out_file: str = '/tmp/profile.pstat', enable=True):
    if not enable:
        yield
        return
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield pr
    except KeyboardInterrupt:
        pass
    finally:
        pr.disable()
        pstats.Stats(pr).sort_stats(key).print_stats(
            lines).dump_stats(out_file)


@contextmanager
def line_profile(*fun, lines: int = 32, enable=True):
    if not enable:
        yield
        return
    lp = LineProfiler()
    lp.enable_by_count()
    for f in fun:
        lp.add_function(f)
    try:
        yield lp
    except KeyboardInterrupt:
        pass
    finally:
        lp.disable_by_count()
        lp.print_stats()


def import_method(s: str) -> Callable:
    """Import a method from a string description.

    Valid formats include:
        * [...package.]module.class.method
        * [...package.]module.method
    """

    method = None
    try:
        # module.method
        module, method = s.rsplit('.', 1)
        method = getattr(importlib.import_module(module), method)
    except ModuleNotFoundError as e:
        # module.class.method
        module, cls, method = s.rsplit('.', 2)
        cls = getattr(importlib.import_module(module), cls)
        if inspect.isclass(cls):
            method = getattr(cls, method)
        else:
            raise ValueError(F'cls {cls} inferred from "{s}" is not a class!')
    if not callable(method):
        raise ValueError(F'method {method} from {s} not callable!')
    return method


class Profiler:
    """
    FIXME: Abusing this class as a namespace...
    Consider implementing __enter__ and __exit__ instead.
    """
    @dataclass
    class Config(ConfigBase):
        # Whether to enable cProfiler.
        cprofile: bool = False
        out_stats_file: str = '/tmp/profile.pstat'

        # List of function names to profile.
        line_profile: Tuple[Union[Callable, str], ...] = ()

    def __init__(self, cfg: Config):
        self._cfg = cfg

    @contextmanager
    def profile(self):
        cfg = self._cfg
        methods = []
        for s in cfg.line_profile:
            if callable(s):
                # Passthrough if already a method.
                # TODO: callable -> inspect.isroutine?
                # which is preferable?
                method = s
            else:
                try:
                    # NOTE: not always the safest idea.
                    # if s in globals()
                    method = eval(s)
                except NameError as e:
                    method = import_method(s)
            methods.append(method)
        with line_profile(*methods,
                          enable=len(cfg.line_profile) > 0) as lp:
            with profile(enable=cfg.cprofile,
                         out_file=cfg.out_stats_file) as pr:
                yield (lp, pr)


def main():
    def x():
        print('x')
    #with profile():
    #    print(1)
    with line_profile(x, x):
        x()


if __name__ == '__main__':
    main()
