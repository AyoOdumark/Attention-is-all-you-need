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
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
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
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
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
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_of_layers, seq_len, num_of_heads, dropout_proba=0.1):
        super(Encoder, self).__init__()
        self.num_of_layers = num_of_layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_of_heads, seq_len, dropout_proba) for _ in range(num_of_layers)])
    
    def forward(self, source_embeddings):
        output = source_embeddings
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
        return output
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_layers, seq_len, num_of_heads, dropout_proba=0.1):
        super(Decoder, self).__init__()
        self.num_of_layers = num_of_layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_of_heads, seq_len, dropout_proba) for _ in range(num_of_layers)])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_output, target_embeddings):
        output = target_embeddings
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(encoder_output, output)
        return nn.functional.log_softmax(self.linear(output), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_layers, seq_len, num_of_heads, dropout_proba=0.1):
        super(Transformer, self).__init__()
        self.embeddings = EmbeddingLayer(vocab_size, embed_dim, seq_len, dropout_proba)
        self.encoder = Encoder(embed_dim, num_of_layers, seq_len, num_of_heads, dropout_proba)
        self.decoder = Decoder(vocab_size, embed_dim, num_of_layers, seq_len, num_of_heads, dropout_proba)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, source, target):
        source_embeddings = self.embeddings(source)
        target_embeddings = self.embeddings(target)
        encoder_output = self.encoder(source_embeddings)
        decoder_output = self.decoder(encoder_output, target_embeddings)
        output = self.linear(decoder_output)
        return output
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

