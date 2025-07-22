#!/usr/bin/env python3

class StridedPatchStem(nn.Module):

    @dataclass
    class Config(ConfigBase):
        c_in: int = -1
        c_out: int = -1
        patch_size: int = -1
        bias: bool = True
        initializer_range: float = 0.02

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_parameter('weight', nn.Parameter(
            th.zeros((cfg.c_in, cfg.c_out, cfg.patch_size, cfg.patch_size),
                     dtype=th.float,
                     requires_grad=True),
            requires_grad=True))

        if cfg.bias:
            self.register_parameter(
                'bias',
                nn.Parameter(
                    th.empty(
                        cfg.c_out,
                        dtype=th.float,
                        requires_grad=True),
                    requires_grad=True))
        else:
            self.register_paramter('bias', None)

        with th.no_grad():
            self.weight.data.normal_(mean=0.0,
                                     std=cfg.initializer_range)
            if self.bias is not None:
                self.bias.data.zero_()

    def global_to_local(self, x: th.Tensor):
        cfg = self.cfg
        P: int = cfg.patch_size
        ni: int = x.shape[-2] // P
        nj: int = x.shape[-1] // P
        x = rearrange(x,
                      '... c (ni bi) (nj bj) -> ... c (bi ni) (bj nj)',
                      ni=ni, nj=nj,
                      bi=P, bj=P)
        return x

    def local_to_global(self, x: th.Tensor):
        cfg = self.cfg
        P: int = cfg.patch_size
        ni: int = x.shape[-2] // P
        nj: int = x.shape[-1] // P
        x = rearrange(x,
                      '... c (bi ni) (bj nj) -> ... c (ni bi) (nj bj)',
                      ni=ni, nj=nj,
                      bi=P, bj=P)
        return x

    def forward(self, x: th.Tensor):
        x = rearrange(x,
                      '... i (p n) (q m) -> ... i p n q m',
                      p=self.cfg.patch_size, q=self.cfg.patch_size)
        x = th.einsum('... ipnqm, iopq -> ... onm',
                      # ci (ph nh) (pw nw), ci co ph pw -> ... co (nh nw)',
                      x, self.weight)
        x = x + self.bias[..., None, None]
        return x

