# %%
import torch
import pdb
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Type, List
import torch.nn.functional as F
from torchvision import transforms

import backbone.SAMEncoder as encoder
from backbone.common import LayerNorm2d
from backbone.transformer import TwoWayTransformer as transformer

from adapter.AdapterModule import AdapterModule, SharedUpLayer
from adapter.ECTune import ECTune
from adapter.PETune import PETune

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
        ec_tuned = self.ECTune(x) # N x (H / patch_size) x (W / patch_size) x embed_dim

        x = self.patch_embed(x) # N x (H / patch_size) x (W / patch_size) x embed_dim
        if self.pos_embed is not None:
            x += self.pos_embed
 
        pe_tuned = self.PETune(x)
        distilled = ec_tuned + pe_tuned # N x (H / patch_size) x (W / patch_size) x embed_dim

        for i, blk in enumerate(self.blocks):
            x = blk(x) # N x (H / patch_size) x (W / patch_size) x embed_dim
            x += self.adapters(distilled, i) 
        x = self.neck(x.permute(0, 3, 1, 2))
        return x # N x out_chans x (H / patch_size) x (W / patch_size)

class WebSAMDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3, # TODO: figure out a good default
        activation: Type[nn.Module] = nn.GELU
    ) -> None:
        """
        SAM mask decoder without prompt encoding and IoU prediction

        Args:
            transformer_dim (int): transformer channel dimension
            transformer (nn.Module): transformer for mask prediction
            num_multimask_outputs (int, optional): number of masks to output
            activation (Type[nn.Module], optional): activation function
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, 32, 3)
                for i in range(self.num_mask_tokens)
            ]
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool = False, # TODO: check if right default
    ) -> torch.Tensor:
        """
        Mask prediction given image embedding

        Args:
            image_embeddings (torch.Tensor): encoded input image
            image_pe (torch.Tensor): positional encoding of input image
            multimask_output (bool, optional): if true, returns multiple masks

        Returns:
            torch.Tensor: batched predicted masks
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        
        masks = masks[:, mask_slice, :, :]
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor
    ) -> torch.Tensor:
        
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(1, -1, -1) # emulating one-prompt shape from promptable SAM
        src = torch.repeat_interleave(image_embeddings, output_tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, output_tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # print(f'src shape: {src.shape}')
        # print(f'pos_src shape: {pos_src.shape}')
        # print(f'output_tokens shape: {output_tokens.shape}')
        
        hs, src = self.transformer(src, pos_src, output_tokens)
        # print(f'hs shape after transformer: {hs.shape}')
        # print(f'src shape after transformer: {src.shape}')
        
        mask_tokens_out = hs[:, :self.num_mask_tokens, :]
        # print(f'mask_tokens_out shape: {mask_tokens_out.shape}')
        
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # print(f'upscaled_embedding shape: {upscaled_embedding.shape}')

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # print(f'hyper_in shape: {hyper_in.shape}')
        
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # print(f'masks shape: {masks.shape}')

        return masks
    
class WebSAMAdapter(nn.Module):
    def __init__(
        self,
        encoder: WebSAMEncoder,
        decoder: WebSAMDecoder
    ) -> None:
        """
        Complete WebSAM Adapter model

        Args:
            encoder (WebSAMEncoder): image encoder
            decoder (WebSAMDecoder): mask decoder
        """
        super().__init__()
        self.encoder = encoder
        self.pe_layer = PositionEmbeddingRandom(128)
        self.decoder = decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.encoder(x)
        mask = self.decoder(image_embeddings, image_pe=self.pe_layer((64, 64)).unsqueeze(0))
        # #TODO: post process back to OG shape
        # mask = self.postprocess_masks(low_res_mask, input_size=(1024, 1024), original_size=original_shape)
        # pdb.set_trace()
        return mask


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
# %%


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C