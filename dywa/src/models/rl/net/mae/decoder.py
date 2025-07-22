#!/usr/bin/env python3
class VITSSDecoder(nn.Module):
    def __init__(self, cfg: ViTMAEConfig):
        super().__init__()
        self.cfg = cfg
        cfg_impl = ViTMAEConfigImpl(
            **asdict(self.cfg))

        # Masking
        self.register_parameter(
            'mask_token', nn.Parameter(
                th.zeros(
                    1, 1, cfg.decoder_hidden_size)))

        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            cfg.decoder_hidden_size,
            int(cfg.image_size // cfg.patch_size),
            add_cls_token=True
        )
        self.register_buffer('pos_embed',
                             th.from_numpy(pos_embed).float(),
                             persistent=False)

        # Decoder
        self.decoder_embed = nn.Linear(
            cfg.hidden_size, cfg.decoder_hidden_size, bias=True)

        decoder_config = deepcopy(cfg_impl)
        decoder_config.hidden_size = cfg_impl.decoder_hidden_size
        decoder_config.num_hidden_layers = cfg_impl.decoder_num_hidden_layers
        decoder_config.num_attention_heads = cfg_impl.decoder_num_attention_heads
        decoder_config.intermediate_size = cfg_impl.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config)
                for _ in range(cfg_impl.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(
            cfg_impl.decoder_hidden_size,
            eps=cfg_impl.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            cfg_impl.decoder_hidden_size,
            cfg_impl.patch_size**2 * cfg_impl.num_channels,
            bias=True
        )  # encoder to decoder

        # Init weights
        nn.init.normal_(self.mask_token,
                        std=cfg_impl.initializer_range)

    def forward(self, x: th.Tensor, ids_restore: th.Tensor) -> th.Tensor:
        """
        x: (..., P , C) feature map.
        """
        cfg = self.cfg
        x = self.decoder_embed(x)

        # append mask token to sequence : Batch X (tokens + cls (1) - non
        # masked tokens) X featrue dim
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = th.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = th.gather(x_,
                       dim=1,
                       index=ids_restore.unsqueeze(-1).repeat(1,
                                                              1,
                                                              x.shape[2]))  # unshuffle
        x = th.cat([x[:, :1, :], x_], dim=1)

        # Add positional embeddings.
        x = x + self.pos_embed

        # Mix.
        for i, layer_module in enumerate(self.decoder_layers):
            layer_outputs = layer_module(x,
                                         head_mask=None,
                                         output_attentions=False)
            x = layer_outputs[0]

        # predict output.
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        # Note: reconstructed fetures not the image itself.

        x = x[:, 1:, :]
        n: int = int(cfg.image_size // cfg.patch_size)
        # B n^2 ph pw c
        x_unpatched = rearrange(
            x,
            '... (nh nw) (ph pw c) -> ... c (nh ph) (nw pw)',
            nh=n,
            nw=n,
            ph=cfg.patch_size,
            pw=cfg.patch_size)
        return x, x_unpatched

