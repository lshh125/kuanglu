import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kuanglu.model_components import (
    CellDenoise,
    CellEmbed,
    CellInteract,
    CellQualify,
    CellReconst,
    CellSmooth,
)
from kuanglu.model_utils import MLP, SelfAttention
from kuanglu.utils import masking


class doubleAELoss(nn.Module):
    def __init__(self, s2r):
        super(doubleAELoss, self).__init__()
        self.s2r = s2r
        
    def forward(self, X, X_hat):
        X_de, Z, Q, X_rec, X_sm = X_hat
        # Reconstruction loss
        loss_rec = F.mse_loss(X_rec, X_de)
        # Smoothness loss
        loss_sm = F.mse_loss(X_sm, X)
        # Total loss
        loss = loss_rec + self.s2r * loss_sm 
        
        return loss, loss_rec, loss_sm


class ModelAEVar(nn.Module):
    
    """Autoencoder-like variant of the original implementation
    
    """
    
    def __init__(self, *, 
                 d_gene: int, 
                 d_denoise: list, 
                 d_quality: list, 
                 lbd: float=1.0, 
                 lbdCI=0.5,
                 embed_config: dict={'embedType': 'transformer', 
                                     'default': True},
                 spatial_config: dict={'n_heads': 1,
                                       'length_scale': 100.},
                 ):
        super(ModelAEVar, self).__init__()
        
        self.cell_denoise = CellDenoise([d_gene] + d_denoise + [d_gene])
        self.cell_embed = CellEmbed(**embed_config)
        embed_dim = self.cell_embed.dummyForward(torch.zeros((1, d_gene))).shape[-1]
        self.cell_reconst = CellReconst(embedType='ff', ldim=[d_gene] + d_denoise + [d_gene])
        self.cell_qualify = CellQualify([embed_dim] + d_quality + [1])
        self.cell_smooth = CellSmooth()
        self.cell_interact = False # Cancelled cell interaction for an Encoder-Decoder structure
        
    def forward(self, X):
        
        X_de = self.cell_denoise(X)
        Z = self.cell_embed(X_de)
        Q = self.cell_qualify(Z)
        X_rec = self.cell_reconst(Z)
        X_sm = self.cell_smooth(X_rec, Z, Q)
        
        return (X_de, Z, Q, X_rec, X_sm)
        
    def getCellEmbedding(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            X = self.cell_denoise(X)
        return self.cell_embed(X)
        
    def fit(self, what, train_loader, validate_loader, epochs, device='cuda',
            cell_masking_rate=0.3, gene_masking_rate=0.6,
            validate_per=1, lr=1e-3, l2_reg=1e-4, fix=None, lassoW='VC', spatial=False
            ):
        pass