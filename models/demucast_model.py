import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from models.AST_model import ASTModel

class DemucastModel(nn.Module):
    def __init__(self, pre_encoder, args):
        super(DemucastModel, self).__init__()
        print('-' * 15 + 'Demucast Model building' + '-' * 15)
        self.encoder = pre_encoder
        self.isbaseline = args.baseline
        self.label_dim = args.label_dim
        self.ast = ASTModel(label_dim=self.label_dim)




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @autocast()
    def forward(self, x):
        if self.isbaseline == False:
            x = self.encoder(x) # shape: (4, 1876, 128)
        else:
            x = x
        B, C, L = x.shape
        x = self.ast(x)

        return x
