import torch
import torch.nn as nn

from .base import Prober


class LinearProbe(Prober):
    def __init__(self, patch_size = 16, embed_dim = 1024, output_dim = 1, dropout = 0.0, init_scales=None, init_shifts=None, *args, **kwargs):
        super().__init__(output_dim=output_dim, init_scales=init_scales, init_shifts=init_shifts)
        
        self.tokens_pframe = int(patch_size ** 2)
        self.probe = nn.ModuleList([
            nn.Linear(embed_dim, 1, bias=True) for _ in range(output_dim)
        ])
        
    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        
        x = x.view(B, N // self.tokens_pframe, self.tokens_pframe, D)
        x = x.mean(2).squeeze(dim = 2)
        
        head_outputs = [head(x) for head in self.probe]
        out = torch.cat(head_outputs, dim=-1)
        return self.apply_output_affine(out)


class NonLinearProbe(Prober):
    def __init__(
        self,
        patch_size=16,
        embed_dim=1024,
        hidden_dim=None,
        output_dim=1,
        dropout=0.0,
        init_scales=None,
        init_shifts=None,
        *args,
        **kwargs,
    ):
        super().__init__(output_dim=output_dim, init_scales=init_scales, init_shifts=init_shifts)

        self.tokens_pframe = int(patch_size ** 2)
        hidden_dim = hidden_dim or (embed_dim // 2)
        self.probe = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=True),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1, bias=True),
            )
            for _ in range(output_dim)
        ])

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape

        x = x.view(B, N // self.tokens_pframe, self.tokens_pframe, D)
        x = x.mean(2).squeeze(dim=2)

        head_outputs = [head(x) for head in self.probe]
        out = torch.cat(head_outputs, dim=-1)
        return self.apply_output_affine(out)
    
    
if __name__ == '__main__':
    device = torch.device('cuda')
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    encoder = encoder.to(device)
    probe = LinearProbe(embed_dim = encoder.embed_dim, output_dim = 2).to(device)
    
    with torch.no_grad(): 
        input = torch.rand((4, 3, 18 + 2, 256, 256)).to(device)
        output = probe(encoder(input)) 
        print(output.shape)