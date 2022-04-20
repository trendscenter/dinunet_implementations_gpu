import torch.nn as nn
from segmenter.segm.model.decoder3d import MaskTransformer3d
from segmenter.segm.model.segmenter3d import Segmenter3d
from segmenter.segm.model.vit3d import VisionTransformer3d

class BrainVit(nn.Module):
    def __init__(self, 
                 n_classes, 
                 patch_size=32, 
                 sv_w=32, 
                 sv_h=32, 
                 sv_d=32, 
                 vit_enc_n_layers=12,
                 vit_enc_d_model = 128,
                 vit_enc_d_ff = 128,
                 vit_enc_n_heads = 8,
                 vit_dec_n_layers = 2,
                 vit_dec_n_heads=8,
                 vit_dec_d_model=128,
                 vit_dec_d_ff=128,
                 drop_path_rate=0.0,
                 dropout=0.0,
                 **kwargs):
        self.vit = VisionTransformer3d((sv_w, sv_h, sv_d), patch_size, vit_enc_n_layers, vit_enc_d_model, vit_enc_d_ff, vit_enc_n_heads, n_classes, channels=1)
        self.decoder = MaskTransformer3d(n_classes, patch_size, vit_enc_d_ff, vit_dec_n_layers, vit_dec_n_heads, vit_dec_d_model, vit_dec_d_ff, drop_path_rate, dropout)
        self.net = Segmenter3d(self.vit, self.decoder, n_cls=n_classes)

    def forward(self, x):
        return self.net(x)