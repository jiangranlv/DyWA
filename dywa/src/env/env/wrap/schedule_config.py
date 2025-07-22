#!/usr/bin/env python3

from typing import Optional, Dict
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

from util.config import ConfigBase, recursive_replace_map
from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv


class ScheduleConfig(WrapperEnv):

    @dataclass
    class Config(ConfigBase):
        schedule_type: str = 'linear'  # or 'logarithmic' or 'none'
        start_value: float = float('nan')
        final_value: float = float('nan')
        total_steps: int = 0

    def __init__(self,
                 cfg: Config,
                 env: EnvIface,
                 key: str,
                 writer: SummaryWriter):
        super().__init__(env)
        self._step_count: int = 0
        self._writer = writer

    def step(self):
        value = self._scheduler(self._step_count)
        # update `cfg` and hope that everything still works afterwards
        self.env.cfg = recursive_replace_map(self.env.cfg,
                                             {self.key: value})


def main():

    @dataclass
    class InnerConfig(ConfigBase):
        a: int = 2
        b: Optional[Dict[str, int]] = None

    @dataclass
    class OuterConfig(ConfigBase):
        inner: InnerConfig = InnerConfig()

    class InnerClass:
        def __init__(self, cfg: InnerConfig):
            self.cfg = cfg

        def __call__(self):
            print('self.cfg', self.cfg)

    cfg = OuterConfig()
    cls = InnerClass(cfg.inner)
    cls()

    # Updated `cfg` >> this does not work
    # cfg = recursive_replace_map(cfg,
    #                             {'inner.b': {'a': 3}})

    # Update in-place >> this _does_ work.
    cfg.inner.b = {'a': 3}
    cls()


if __name__ == '__main__':
    main()
