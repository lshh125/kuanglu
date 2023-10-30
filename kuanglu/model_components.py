import torch
import torch.nn as nn

from kuanglu.model_utils import DecoderBlock, EncoderBlock


class CellEmbed(nn.Module):
    def __init__(self, embedType='transformer', **kwargs):
        """Generate cell Embedding

        :param embedType: 'transformer' or 'ff'
        :param kwargs: other parameters for the embedding, 'ldim' for ff, 
            n_layers, n_heads, d_model, d_ff, dropout for transformer, default to 2, 4, 128, 512, 0.1
        """
        super(CellEmbed, self).__init__()
        
        assert embedType in ['transformer', 'ff'], "Unknown embedding type."
        self.embedType = embedType
        
        if embedType == 'transformer':
            if kwargs['default'] == False:
                assert 'n_layers' in kwargs, "Must provide n_layers for transformer embedding."
                assert 'n_heads' in kwargs, "Must provide n_heads for transformer embedding."
                assert 'd_input' in kwargs, "Must provide d_input for transformer embedding."
                assert 'd_model' in kwargs, "Must provide d_model for transformer embedding."
                assert 'd_ff' in kwargs, "Must provide d_ff for transformer embedding."
                assert 'dropout' in kwargs, "Must provide dropout for transformer embedding."
                n_layers = kwargs['n_layers']
                n_heads = kwargs['n_heads']
                d_input = kwargs['d_input']
                d_model = kwargs['d_model']
                d_ff = kwargs['d_ff']
                dropout = kwargs['dropout']
            n_layers = 2 if 'n_layers' not in kwargs else kwargs['n_layers']
            n_heads = 4 if 'n_heads' not in kwargs else kwargs['n_heads']
            d_input = 128 if 'd_input' not in kwargs else kwargs['d_input']
            d_model = 128 if 'd_model' not in kwargs else kwargs['d_model']
            d_ff = 512 if 'd_ff' not in kwargs else kwargs['d_ff']
            dropout = 0.1 if 'dropout' not in kwargs else kwargs['dropout']
            
            self.net = nn.Sequential(*([nn.Linear(d_input, d_model)] + 
                                       [EncoderBlock(d_model, n_heads, dropout, d_ff) for _ in range(n_layers)]))
            
        elif embedType == 'ff':
            
            #!!!
            #TODO: Rebase this class on MLP in model_utils.py
            
            assert 'ldim' in kwargs, "Must provide ldim for ff embedding."
            d = kwargs['ldim']
            layers = []
            for i in range(len(d) - 1):
                layers.append(nn.Linear(d[i], d[i + 1]))
                if i < len(d) - 2:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)
            
        
    def forward(self, x):
        return self.net(x)
    
    def resetEmbedding(self, init_method='kaiming_normal', **kwargs):
        assert init_method in ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'], "Unknown init method."
        if init_method == 'kaiming_normal':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
        elif init_method == 'kaiming_uniform':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)
        elif init_method == 'xavier_normal':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
        elif init_method == 'xavier_uniform':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    
    def dummyForward(self, x):
        dummyInput = torch.zeros((1,) + x.shape[1:]).to(x.device)
        return self.net(dummyInput)


class CellDenoise(nn.Module):
    
    #!!!
    #TODO: Rebase this class on MLP in model_utils.py
    #! Also need to reconsider the role of dropout here.
    
    def __init__(self, d):
        """Denoise cells

        :param d: list of dimensionality of layers. The first and last number must be the number of genes.
        """
        super(CellDenoise, self).__init__()
        layers = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            if i < len(d) - 2:
                layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x) + x


class CellQualify(nn.Module):
    def __init__(self, d):
        """Calculate a quality score for each cell

        :param d: list of dimensionality of layers. The first number must be the number of genes. The last number
        must be 1.
        """
        super(CellQualify, self).__init__()
        layers = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            if i < len(d) - 2:
                layers.append(nn.Tanh())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class CellSmooth(nn.Module):
    def __init__(self):
        """Smooth across cells but within each gene

        """
        super(CellSmooth, self).__init__()
        self.normalize = torch.nn.Softmax(dim=-1)

    def forward(self, expression, encoding, quality):
        similarity = self.normalize(-torch.cdist(encoding, encoding) + quality.transpose(-1, -2))
        return similarity @ expression
        # return similarity


class CellInteract(nn.Module):
    def __init__(self, d_gene, d_embed, length_scale=100., symmetric=False):
        """Model cell-cell interaction by transform the gene expression across both cells and genes.

        :param d_gene: number of genes
        :param d_embed: dimensionality of input cell embedding
        """
        super(CellInteract, self).__init__()
        self.gene_response = nn.Parameter(torch.randn((d_gene, d_gene)))
        self.transform = nn.Parameter(torch.randn((d_embed, d_embed)))
        self.scale = torch.nn.Sigmoid()
        self.length_scale = nn.Parameter(torch.tensor(length_scale), requires_grad=False)
        # self.lr_mask = torch.tensor(lr_mask, dtype=torch.float32).to(device)
        self.lasso_reg = None
        self.symmetric = symmetric

    def resetCellInteraction(self, init_method='kaiming_normal', **kwargs):
        assert init_method in ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'], "Unknown init method."
        if init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.transform)
            nn.init.kaiming_normal_(self.gene_response)
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.transform)
            nn.init.kaiming_uniform_(self.gene_response)
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.transform)
            nn.init.xavier_normal_(self.gene_response)
        elif init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.transform)
            nn.init.xavier_uniform_(self.gene_response)

    def forward(self, expression, encoding, sqr_pdist=None):
        cell_interaction = self.scale(encoding @ self.transform @ encoding.transpose(-1, -2))
        if sqr_pdist is not None:
            spatial_scaling = torch.exp(- sqr_pdist / (self.length_scale ** 2))
            if self.symmetric:
                return (spatial_scaling * cell_interaction) @ expression @ ((self.gene_response + self.gene_response.T) / 2) / expression.shape[1]
            else:
                return (spatial_scaling * cell_interaction) @ expression @ (self.gene_response) / expression.shape[1]
        else:
            if self.symmetric:
                return cell_interaction @ expression @ ((self.gene_response + self.gene_response.T) / 2) / expression.shape[1]
            else:
                return cell_interaction @ expression @ (self.gene_response) / expression.shape[1]

    def getLassoReg(self, type='V'):
        assert type in ['V', 'C', 'VC'], "Undefined param of lasso regularization."
        self.lasso_reg = torch.tensor(0., dtype=torch.float32).to(self.gene_response.device)
        type = [char for char in type]
        if 'V' in type:
            if self.symmetric:
                self.lasso_reg = self.lasso_reg + torch.sum(torch.abs((self.gene_response + self.gene_response.T) / 2))
            else:
                self.lasso_reg = self.lasso_reg + torch.sum(torch.abs(self.gene_response))
        if 'C' in type:
            self.lasso_reg = self.lasso_reg + torch.sum(torch.abs(self.transform))
        return self.lasso_reg


class CellReconst(nn.Module):
    def __init__(self, embedType='ff', **kwargs):
        """Generate cell Decoding

        :param embedType: 'transformer' or 'ff'
        :param kwargs: other parameters for the decoding, 'ldim' for ff, 
            n_layers, n_heads, d_model, d_ff, dropout for transformer, default to 2, 4, 128, 512, 0.1
        """
        super(CellReconst, self).__init__()
        
        assert embedType in ['transformer', 'ff'], "Unknown decoding type."
        self.embedType = embedType
        
        if embedType == 'transformer':
            n_layers = kwargs.get('n_layers', 2)
            n_heads = kwargs.get('n_heads', 4)
            d_input = kwargs.get('d_input', 128)
            d_model = kwargs.get('d_model', 128)
            d_ff = kwargs.get('d_ff', 512)
            dropout = kwargs.get('dropout', 0.1)
            
            self.net = nn.Sequential(*([DecoderBlock(d_model, n_heads, dropout, d_ff) for _ in range(n_layers)] + 
                                       [nn.Linear(d_model, d_input)]))
            
        elif embedType == 'ff':
            assert 'ldim' in kwargs, "Must provide ldim for ff decoding."
            d = kwargs['ldim']
            layers = []
            for i in range(len(d) - 1, 0, -1):
                layers.append(nn.Linear(d[i], d[i - 1]))
                if i > 1:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.net(x)
    
    def resetDecoding(self, init_method='kaiming_normal', **kwargs):
        assert init_method in ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'], "Unknown init method."
        if init_method == 'kaiming_normal':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
        elif init_method == 'kaiming_uniform':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)
        elif init_method == 'xavier_normal':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
        elif init_method == 'xavier_uniform':
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    
    def dummyForward(self, z):
        dummyInput = torch.zeros((1,) + z.shape[1:]).to(z.device)
        return self.net(dummyInput)


