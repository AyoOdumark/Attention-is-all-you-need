import torch
import torch.nn as nn
from attention import MultiHeadAttention

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, dropout_probability=0.1):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)
        
    def forward(self, input_ids):
        _, seq_len = input_ids.shape
        word_embeddings = self.word_embedding(input_ids)
        
        positional_embeddings = self.positional_embedding(torch.arange(seq_len))
        embeddings = word_embeddings + positional_embeddings
        return self.dropout(embeddings)
    
class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, model_dim, width_factor=4):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.linear_1 = nn.Linear(model_dim, width_factor*model_dim)
        self.linear_2 = nn.Linear(width_factor*model_dim, model_dim)
        self.relu = nn.ReLU()
    
    def forward(self, input_tensors):
        output = self.linear_1(input_tensors)
        output = self.relu(output)
        output = self.linear_2(output)
        
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout_probability=0.1):
        super(EncoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f"model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}"
        self.head_dim = int(model_dim / num_of_heads)
        
        self.multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_length, dropout_probability, masked=False)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.position_wise_feed_forward = PositionWiseFeedForwardNet(model_dim)
        self.dropout = nn.Dropout(dropout_probability)
        
    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim)
        return W(X)
        
    def forward(self, input_tensors):
        attention_vectors = self.multi_head_attention(
                                                    self._linear_projection(input_tensors), 
                                                    self._linear_projection(input_tensors), 
                                                    self._linear_projection(input_tensors))
        
        attention_vectors = self.dropout(attention_vectors)
        attention_vectors = self.layer_norm_1(attention_vectors + input_tensors)
        
        feed_forward_output = self.position_wise_feed_forward(attention_vectors)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.layer_norm_2(feed_forward_output + attention_vectors)
        
        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout_probability=0.1):
        super(DecoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f"model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}"
        self.head_dim = int(model_dim / num_of_heads)
        
        self.masked_multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_length, dropout_probability, masked=True)
        self.multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_length, dropout_probability, masked=False)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.layer_norm_3 = nn.LayerNorm(model_dim)
        self.position_wise_feed_forward = PositionWiseFeedForwardNet(model_dim)
        self.dropout = nn.Dropout(dropout_probability)
        
    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim)
        return W(X)
    
    def forward(self, encoder_output, decoder_input):
        keys, queries, values = self._linear_projection(decoder_input), self._linear_projection(decoder_input), self._linear_projection(decoder_input)
        masked_attention_vectors = self.masked_multi_head_attention(keys, queries, values)
        
        masked_attention_vectors = self.dropout(masked_attention_vectors)
        masked_attention_vectors = self.layer_norm_1(masked_attention_vectors + decoder_input)
        
        attention_vectors = self.multi_head_attention(self._linear_projection(encoder_output), queries, self._linear_projection(encoder_output))
        attention_vectors = self.dropout(attention_vectors)
        attention_vectors = self.layer_norm_2(attention_vectors + masked_attention_vectors)
        
        feed_forward_output = self.position_wise_feed_forward(attention_vectors)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.layer_norm_3(feed_forward_output + attention_vectors)
        
        return output
    

