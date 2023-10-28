import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.dropout = dropout
        
        for i in range(len(hidden_size_list)):
            if i == 0:
                setattr(self, "fc{}".format(i), nn.Linear(input_size, hidden_size_list[i]))
            else:
                setattr(self, "fc{}".format(i), nn.Linear(hidden_size_list[i-1], hidden_size_list[i]))
                
        self.fc_out = nn.Linear(hidden_size_list[-1], output_size)
        
    def forward(self, x):
        for i in range(len(self.hidden_size_list)):
            x = getattr(self, "fc{}".format(i))(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            
        x = self.fc_out(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Q, K, V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # output
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        
        N = queries.shape[0]

        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = queries.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, -1, self.heads * self.head_dim
        )
        
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        
        return out
    

class FeedForward(nn.Module):
    def __init__(self, embed_size, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, d_ff):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.layer_norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.layer_norm2(forward + x))
        return out
    

class EncoderBlock(TransformerBlock):
    def __init__(self, embed_size, heads, dropout, d_ff):
        super(EncoderBlock, self).__init__(embed_size, heads, dropout, d_ff)
        

    def forward(self, x, mask=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.layer_norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.layer_norm2(forward + x))
        return out


class AutoEncoderMLP(nn.Module):
    def __init__(self, encoding_dim_list, decoding_dim_list, dropout):
        """

        Args:
            encoding_dim_list (list): input_dim, hidden_dim(s), latent_dim
            decoding_dim_list (list): latent_dim, hidden_dim(s), output_dim
            dropout (float): dropout rate
        """
        super(AutoEncoderMLP, self).__init__()
        self.encoding_dim_list = encoding_dim_list
        self.decoding_dim_list = decoding_dim_list
        self.dropout = dropout
        
        self.encoder = MLP(encoding_dim_list[0], encoding_dim_list[1:-1], encoding_dim_list[-1], dropout)
        self.decoder = MLP(decoding_dim_list[0], decoding_dim_list[1:-1], decoding_dim_list[-1], dropout)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        lat = self.encoder(x)
        return lat
    
    def decode(self, lat):
        x = self.decoder(lat)
        return x


