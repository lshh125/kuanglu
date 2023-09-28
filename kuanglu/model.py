import warnings

import torch
from torch import nn, optim


class CellEmbed(nn.Module):
    def __init__(self, d):
        """Generate cell Embedding

        :param d: list of dimensionality of layers. The first number must be the number of genes.
        """
        super(CellEmbed, self).__init__()
        layers = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            if i < len(d) - 2:
                layers.append(nn.Tanh())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class CellDenoise(nn.Module):
    def __init__(self, d):
        """Denoise cells

        :param d: list of dimensionality of layers. The first and last number must be the number of genes.
        """
        super(CellDenoise, self).__init__()
        layers = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            if i < len(d) - 2:
                layers.append(nn.Tanh())
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
    def __init__(self, d_gene, d_embed):
        """Model cell-cell interaction by transform the gene expression across both cells and genes.

        :param d_gene: number of genes
        :param d_embed: dimensionality of input cell embedding
        """
        super(CellInteract, self).__init__()
        self.gene_response = nn.Parameter(torch.zeros((d_gene, d_gene)))
        self.transform = nn.Parameter(torch.zeros((d_embed, d_embed)))
        self.scale = torch.nn.Sigmoid()
        # self.lr_mask = torch.tensor(lr_mask, dtype=torch.float32).to(device)
        self.lasso_reg = None

    def resetCellInteraction(self, init_method='kaiming_normal', **kwargs):
        assert init_method in ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'], "Unknown init method."
        if init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.transform)
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.transform)
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.transform)
        elif init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.transform)

    def forward(self, expression, encoding):
        cell_interaction = self.scale(encoding @ self.transform @ encoding.transpose(-1, -2))
        
        return cell_interaction @ expression @ (self.gene_response) / expression.shape[1]

    def getLassoReg(self):
        self.lasso_reg = torch.sum(torch.abs(self.transform))
        return self.lasso_reg


class Model(nn.Module):
    def __init__(self, *, d_gene: int, d_denoise: list, d_embed: list, d_quality: list, n_heads: int, lbd: float=1.0, lbdCI=0.5):
        """The entire model

        :param d_gene: number of genes
        :param d_denoise: dims of denoise layers (excluding the first and last one, which are always d_gene)
        :param d_embed: dims of embedding layers (excluding the first one, d_gene)
        :param d_quality: dims of qualify layers (excluding the first one, d_gene, and the last one, 1)
        :param n_heads: number of heads for cell-cell interaction modeling
        """
        super(Model, self).__init__()
        self.cell_embed = CellEmbed([d_gene] + d_embed)
        self.cell_qualify = CellQualify([d_gene] + d_quality + [1])
        self.cell_denoise = CellDenoise([d_gene] + d_denoise + [d_gene])
        self.cell_smooth = CellSmooth()
        self.cell_interacts = nn.ModuleList([CellInteract(d_gene, d_embed[-1]) for i in range(n_heads)])
        for moduleCI in self.cell_interacts:
            moduleCI.resetCellInteraction(init_method='xavier_normal')

        self.lbd = lbd
        self.lbdCI = lbdCI

    def forward(self, raw_expr, interact=False):
        denoised_expr = self.cell_denoise(raw_expr)

        cell_quality = self.cell_qualify(raw_expr)
        cell_embedding = self.cell_embed(raw_expr)

        smoothed_expr = self.cell_smooth(denoised_expr, cell_embedding, cell_quality)

        if interact:
            final_expr = smoothed_expr + self.lbd * sum(f(smoothed_expr, cell_embedding) for f in self.cell_interacts)
            return denoised_expr, smoothed_expr, final_expr
        else:
            return denoised_expr, smoothed_expr

    def fit(self, what, train_loader, validate_loader, epochs, device='cuda',
            cell_masking_rate=0.3, gene_masking_rate=0.6,
            validate_per=1, lr=1e-3, l2_reg=1e-4, fix=None):
        """Fit the model

        :param what: train which part of the network? Either 'denoised', 'smoothed', or 'final'
        :param train_loader: DataLoader for training data
        :param validate_loader: DataLoader for validation data
        :param epochs: number (integer) or range (list of two integers) of epochs
        :param device: 'cpu' or 'cuda'; it needs to match where the model is; it does not automatically move the model
        :param cell_masking_rate: how many cells to mask for training
        :param gene_masking_rate: how many genes to mask for training
        :param validate_per: run validation every ? iterations
        :param lr: learning rate
        :param l2_reg: strength of L2 regularization (aka weight_decay)
        :param fix: A list of modules to fix during fitting. Choose from 'denoise', 'embed', 'qualify', 'smooth', and 'interact'
        :return: record of losses
        """
        def masking(X, *, cell_rate=.6, gene_rate=.6, copy=True):
            if copy:
                X = X.clone()
            row_mask = (torch.rand((X.shape[-2], 1)) < cell_rate) + 0.
            col_mask = (torch.rand((1, X.shape[-1])) < gene_rate) + 0.
            mask = 1 - (row_mask * col_mask).to(device)
            return X * mask, row_mask.squeeze(), col_mask.squeeze()

        whats = ['denoised', 'smoothed', 'final']
        if what in whats:
            what = whats.index(what)
        else:
            raise ValueError("Can only train for ['denoised', 'smoothed', or 'final'.")

        if isinstance(epochs, int):
            epochs = [0, epochs]
        elif isinstance(epochs, list):
            assert (len(epochs) == 2)
            assert (isinstance(epochs[0], int) and isinstance(epochs[1], int))
        else:
            raise ValueError("epochs must be an integer or a list of two numbers.")

        fixed_modules = []
        if fix is None:
            fix = []
        for i in fix:
            if i == 'denoise':
                fixed_modules.append(self.cell_denoise)
            elif i == 'embed':
                fixed_modules.append(self.cell_embed)
            elif i == 'qualify':
                fixed_modules.append(self.cell_qualify)
            elif i == 'smooth':
                fixed_modules.append(self.cell_smooth)
            elif i == 'interact':
                fixed_modules.append(self.cell_interacts)
            else:
                raise ValueError("Unknown module to be fixed during training.")

        for module in fixed_modules:
            for param in module.parameters():
                if not param.requires_grad:
                    warnings.warn("Some parameters are already fixed. They will be unfixed when exiting this function.")
                param.requires_grad = False

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        regression_criterion = nn.MSELoss()

        train_epoch = []
        train_mse = []
        validate_epoch = []
        validate_mse = []
        raw_mse = []

        for epoch in range(*epochs):
            temp_mse = 0.
            for X in train_loader:
                X = X.to(device)
                X2, cell_mask, gene_mask = masking(X, cell_rate=cell_masking_rate, gene_rate=gene_masking_rate)

                res = self(X2, what != 'final')[what]
                regression_loss = regression_criterion(res[:, cell_mask == 1, :][:, :, gene_mask == 1],
                                                       X[:, cell_mask == 1, :][:, :, gene_mask == 1])
                loss_optim = regression_loss
                for headCI in self.cell_interacts:
                    loss_optim += self.lbdCI * headCI.getLassoReg()

                optimizer.zero_grad()
                loss_optim.backward()
                optimizer.step()

                temp_mse += regression_loss.detach().to('cpu').item()

            train_epoch.append(epoch)
            train_mse.append(temp_mse / len(train_loader))
            if epoch == 0 or (epoch + 1) % validate_per == 0:
                temp_mse = 0.
                with torch.no_grad():
                    for X in validate_loader:
                        X = X.to(device)
                        X2, cell_mask, gene_mask = masking(X, cell_rate=cell_masking_rate, gene_rate=gene_masking_rate)
                        raw_mse.append(regression_criterion(X2[:, cell_mask == 1, :][:, :, gene_mask == 1],
                                                            X[:, cell_mask == 1, :][:, :, gene_mask == 1]).to('cpu').item())
                        res = self(X2, what != 'final')[what]
                        regression_loss = regression_criterion(res[:, cell_mask == 1, :][:, :, gene_mask == 1],
                                                               X[:, cell_mask == 1, :][:, :, gene_mask == 1])

                        temp_mse += regression_loss.detach().to('cpu').item()

                validate_epoch.append(epoch)
                validate_mse.append(temp_mse / len(validate_loader))

                print('Epoch', '%04d' % (epoch + 1),
                      'Train MSE', '{:.3f}'.format(train_mse[-1]),
                      'Validate MSE', '{:.3f}'.format(validate_mse[-1]),
                      'Raw MSE', '{:.3f}'.format(raw_mse[-1]))

        for module in fixed_modules:
            for param in module.parameters():
                param.requires_grad = True

        return {'train_epoch': train_epoch, 'train_mse': train_mse,
                'validate_epoch': validate_epoch, 'validate_mse': validate_mse, 'raw_mse': raw_mse}
