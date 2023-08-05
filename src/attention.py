import torch 
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, seq_length, dropout_probability, masked=True):
        super(Attention, self).__init__()
        self.masked = masked
        
        if self.masked:
            self.register_buffer("tril", torch.tril(torch.ones(seq_length, seq_length)))
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout_probability)
        
    
    def forward(self, keys, queries, values, head_dim):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim).float())
        
        if self.masked:
            scores = scores.masked_fill(self.tril==0, float("-inf"))
            
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        
        attention_vectors = torch.matmul(attention_weights, values)
        return attention_vectors
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout_probability, masked=True):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_of_heads == 0, f"model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}"
        
        self.head_dim = int(model_dim / num_of_heads)
        self.W_q = nn.Linear(model_dim, self.head_dim)
        self.W_k = nn.Linear(model_dim, self.head_dim)
        self.W_v = nn.Linear(model_dim, self.head_dim)
    
        self.attention_heads = nn.ModuleList(Attention(seq_length, dropout_probability, masked) 
                                             for _ in range(num_of_heads))
        self.W_o = nn.Linear(num_of_heads*self.head_dim, model_dim)
        self.dropout = nn.Dropout(dropout_probability)
        
    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim)
        return W(X)
        
    def forward(self, keys, queries, values):
        heads = [attention_head(self._linear_projection(keys), 
                                    self._linear_projection(queries), 
                                    self._linear_projection(values), 
                                    self.head_dim
                                )
                    for attention_head in self.attention_heads
                ]
            
        concatenated_heads = torch.cat(heads, dim=-1)
        attention_vectors = self.W_o(concatenated_heads)
        attention_vectors = self.dropout(attention_vectors)
        
        return attention_vectors
