from adapter.AdapterModule import AdapterModule, SharedUpLayer
from adapter.ECTune import ECTune
from adapter.PETune import PETune
import backbone.SAMEncoder as encoder
from backbone.SAMDecoder import MaskDecoder as decoder
from backbone.common import LayerNorm2d

import torch
import torch.nn as nn
from typing import Optional, Tuple, Type

class WebSAMEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = False,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Maps image to webpage-masking space

        Args:
            img_size (int, optional): input image size
            patch_size (int, optional): patch size
            in_chans (int, optional): number of input color channels
            embed_dim (int, optional): transformer / adapter block embedding dimension
            depth (int, optional): number of transformer / adapter blocks
            num_heads (int, optional): number of attention heads
            mlp_ratio (float, optional): mlp hidden layer ratio
            out_chans (int, optional): output channel dimension
            qkv_bias (bool, optional): if true, use bias in qkv projection
            norm_layer (Type[nn.Module], optional): normalization layer
            act_layer (Type[nn.Module], optional): activation layer
            use_abs_pos (bool, optional): if true, use absolute position embeddings
            use_rel_pos (bool, optional): if true, add relative positional embeddings to the attention map
            rel_pos_zero_init (bool, optional): if true, initialize relative positional embeddings to zero
            window_size (int, optional): _description_. window size for window attention blocks
            global_attn_indexes (Tuple[int, ...], optional): indices for blocks using global attention
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = encoder.PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = encoder.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.ECTune = ECTune(embed_dim=embed_dim, patch_size=patch_size)
        self.PETune = PETune(embed_dim=embed_dim)
        self.shared = SharedUpLayer(input_size = embed_dim, num_layers = 1)
        self.adapters = AdapterModule(shared = self.shared, num_adapters = depth, num_layers = 1)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x # N x C x H x W
        x = self.patch_embed(x) # N x (H / patch_size) x (W / patch_size) x embed_dim
        if self.pos_embed is not None:
            x += self.pos_embed

        ec_tuned = self.ECTune(input)
        pe_tuned = self.PETune(x)
        distilled = ec_tuned + pe_tuned # N x (H / patch_size) x (W / patch_size) x embed_dim

        for i, blk in enumerate(self.blocks):
            x = blk(x) # N x (H / patch_size) x (W / patch_size) x embed_dim
            x += self.adapters(distilled, i) 
        x = self.neck(x.permute(0, 3, 1, 2))
        return x # N x out_chans x (H / patch_size) x (W / patch_size)
