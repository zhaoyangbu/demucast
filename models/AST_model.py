import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import timm
from einops import rearrange


class ASTModel(nn.Module):
    def __init__(self, f_stride=10, t_stride=10, f_dim=128, t_dim=1876, label_dim=10, vit_model='tiny224',
                 vit_pretrain=True):
        super(ASTModel, self).__init__()


        if vit_model not in ['tiny224', 'small224', 'base224', 'base384']:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.', )

        self.vit = timm.create_model(f'deit_{vit_model[:-3]}_distilled_patch16_{vit_model[-3:]}',
                                     pretrained=vit_pretrain)

        self.vit_num_patches = self.vit.patch_embed.num_patches
        self.vit_patch_n_hw = self.vit.patch_embed.grid_size
        self.vit_embedding_dim = self.vit.pos_embed.shape[2]

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.vit_embedding_dim),
                                      nn.Linear(self.vit_embedding_dim, label_dim))

        self.f_input_size = int((f_dim - 1 * (16 - 1) - 1) / f_stride + 1)
        self.t_input_size = int((t_dim - 1 * (16 - 1) - 1) / t_stride + 1)
        self.vit.f_input_size = self.f_input_size
        self.vit.t_input_size = self.t_input_size
        # average the pretrain vit embedding layers and make the input to 1 channel
        new_proj = torch.nn.Conv2d(1, self.vit_embedding_dim, kernel_size=(16, 16), stride=(f_stride, t_stride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.vit.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.vit.patch_embed.proj.bias
        self.vit.patch_embed.proj = new_proj
        self.vit.patch_embed.img_size = (f_dim, t_dim)

        # build the new pose embedding from the pretrain vit
        pretrained_pos_embed = self.vit.pos_embed[:, :, :].detach()
        pos_embed = rearrange(pretrained_pos_embed[:, 2:, :], '1 (h w) e -> 1 e h w ', h=self.vit_patch_n_hw[0])


        new_pos_embed = F.interpolate(pos_embed, (self.f_input_size, self.t_input_size), mode='bilinear',
                                      align_corners=False)

        new_pos_embed = rearrange(new_pos_embed, '1 e h w -> 1 (h w) e')
        self.vit.original_pose_embed = self.vit.pos_embed
        self.vit.pos_embed = nn.Parameter(torch.cat([pretrained_pos_embed[:, :2, :], new_pos_embed], dim=1))


    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1876, 128)
        :return: prediction
        """
        x = x.real
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        dist_token = self.vit.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    input_tdim = 1876
    ast_mdl = ASTModel(t_dim=input_tdim)
    test_input = torch.rand([1, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)
    # output should be in shape [1, 10], i.e., 10 samples, each with prediction of 10 classes.

