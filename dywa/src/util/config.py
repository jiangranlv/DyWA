#!/usr/bin/env python3

import sys
from functools import wraps
from typing import (
    Callable, Iterable, Mapping,
    Dict, Tuple, List,
    Any, Union,
    Type, TypeVar,
    get_args,
    get_origin
)
from dataclasses import dataclass, replace, is_dataclass, fields
from simple_parsing import Serializable
from omegaconf import OmegaConf
import inspect
from pathlib import Path

# ConfigBase = Serializable
# @dataclass(init=False)
# class ConfigBase:
#     def __init__(self, **kwds):
#         names = set([f.name for f in fields(self)])
#         for k, v in kwds.items():
#             if k in names:
#                 setattr(self, k, v)
ConfigBase = object
D = TypeVar('D')


def dc_from_oc(cls, oc):
    """
    Create dataclass instance based on omegaconf ref.
    """
    # `oc` is already of the target type.
    if isinstance(oc, cls):
        return cls

    # Non-dataset classes; one of the following:
    # > Union
    # > Mapping
    # > Iterable
    if not is_dataclass(cls):

        # Deal with Union[A,None] -> A
        # (TBH, only handles~ Optional[A])
        if get_origin(cls) is Union:
            clss = list(cls.__args__)
            clss.remove(type(None))
            if len(clss) != 1:
                raise ValueError(
                    F'Non-unique possible class = {clss}'
                )
            cls = clss[0]

        # Deal with GenericAlias ...
        type_check = cls
        o_cls = get_origin(cls)
        if o_cls is not None:
            type_check = o_cls

        if issubclass(type_check, Mapping):
            # mapping
            pass

        if issubclass(type_check, Iterable):
            # itreable
            pass

        # TODO: deal with Dict[key, dataclass] <-> Dict[key, Dict[...]]
        if isinstance(cls, Mapping):  # Dict[k,v]
            # Get args from type annotation...

            v_cls = get_args(cls.__annotations__)[1]
            d = {}
            for k, v in oc.items():
                d[k] = dc_from_oc(v)
            return cls(**d)

        return oc

    d = {}
    for f in fields(cls):
        if not f.init:
            continue
        d[f.name] = dc_from_oc(f.type, getattr(oc, f.name))
    return cls(**d)


def with_oc_cli(cls: Union[Type[D], D] = None, argv: List[str] = None):
    """Decorator for automatically adding parsed args from cli to entry
    point."""

    if argv is None:
        argv = sys.argv[1:]

    main = None
    if cls is None:
        # @with_cli()
        need_cls = True
    else:
        if callable(cls) and not is_dataclass(cls):
            # @with_cli
            main = cls
            need_cls = True
        else:
            # @with_cli(cls=Config, ...)
            need_cls = (cls is None)  # FIXME: always False.

    def decorator(main: Callable[[D], None]):
        # NOTE:
        # if `cls` is None, try to infer them from `main` signature.
        inner_cls = cls
        if need_cls:
            sig = inspect.signature(main)
            if len(sig.parameters) == 1:
                key = next(iter(sig.parameters))
                inner_cls = sig.parameters[key].annotation
            else:
                raise ValueError(
                    '#arg != 1 in main {}: Cannot infer param type.'
                    .format(sig))

        # If supplied, load from file.
        # The arguments provided through the file will be
        # overridden by any CLI args if present.
        instance = None

        # NOTE: using @wraps to forward main() documentation.
        @wraps(main)
        def wrapper():
            doc = getattr(main, '__doc__', '')

            default_cfg = OmegaConf.structured(inner_cls)
            OmegaConf.set_struct(default_cfg, False)

            # Then load from CLI.
            cli_cfg = OmegaConf.from_cli(argv)

            # Then load from config files.
            file_cfgs = []
            if 'config_path' in cli_cfg:
                if isinstance(cli_cfg['config_path'], str):
                    file_cfgs = [OmegaConf.load(
                        Path(cli_cfg['config_path']).resolve())]
                else:
                    file_cfgs = [OmegaConf.load(Path(p).resolve())
                                 for p in cli_cfg['config_path']]

            cfgs = [default_cfg] + file_cfgs + [cli_cfg]
            # Then merge all cfgs & call main.
            cfg = dc_from_oc(inner_cls, OmegaConf.merge(*cfgs))
            return main(cfg)
        return wrapper

    if main is not None:
        return decorator(main)
    else:
        return decorator


def recursive_replace(src: dataclass, **changes):
    rchanges = {}
    for k, v in changes.items():
        v0 = getattr(src, k)
        if (is_dataclass(v0) and
                isinstance(v, Mapping)):
            v2 = recursive_replace(v0, **v)
        else:
            v2 = v
        rchanges[k] = v2
    return replace(src, **rchanges)


def recursive_replace_str(src: dataclass,
                          key: str, value: Any):
    """ replace() support for x.y = z """
    keys = key.split('.', maxsplit=1)
    if len(keys) == 1:
        try:
            return replace(src, **{key: value})
        except Exception:
            print(F'replace failed for key={key}')
            raise
    replaced = recursive_replace_str(getattr(src, keys[0]),
                                     keys[1], value)
    return replace(src, **{keys[0]: replaced})


def recursive_replace_strs(
        src: dataclass, *entries: Iterable[Tuple[str, Any]]):
    out = src
    for e in entries:
        out = recursive_replace_str(out, e[0], e[1])
    return out


def recursive_replace_map(
        src: dataclass, entries: Mapping[str, Any]):
    out = src
    for k, v in entries.items():
        out = recursive_replace_str(out, k, v)
    return out


def to_flat_dict(d: dataclass, prefix: str = '',
                 cls: Iterable = None) -> Dict[str, Any]:
    """ Convert dataclass to flat dictionary. """
    out = {}
    for f in fields(d):
        key = F'{prefix}/{f.name}'
        v = getattr(d, f.name)
        # if isinstance(v, Mapping):
        #     for k, v in v.items():
        #         out.update(to_flat_dict(v, key, cls=cls))
        if is_dataclass(v):
            out.update(to_flat_dict(v, key, cls=cls))
        else:
            if (cls is not None):
                if isinstance(v, cls):
                    out[key] = v
            else:
                out[key] = v
    return out


def main():
    @dataclass
    class X:
        x: int = 1

    @dataclass
    class Y:
        y: int = 2

    @dataclass
    class Z:
        x: X = X()
    z = Z()
    z2 = recursive_replace(z, x={'x': 2})
    z3 = recursive_replace_str(z, 'x.x', 3)
    z4 = recursive_replace_str(z, 'x', X(x=4))
    # z4 = recursive_replace_str(z, 'x.x.y', 5)
    print(z)
    print(z2)
    print(z3)
    print(z4)
    # print(z4)


if __name__ == '__main__':
    main()
