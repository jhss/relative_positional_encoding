import tensorflow as tf
import numpy as np

from utils import *

# 오차가 있긴 한데, 이정도는 괜찮음
class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, seq_len, emb_dim):
    super(PositionalEncoding, self).__init__()
    self.seq_len = seq_len
    self.emb_dim = emb_dim

  def get_angle(self):
    seq_idx = np.arange(self.seq_len)[:, np.newaxis]
    dim_idx = np.arange(self.emb_dim)[np.newaxis, :]

    return (1 / np.power(10000, 2 * (dim_idx // 2) / self.emb_dim)) * seq_idx

  def positional_encoding(self):
    pos = np.zeros(shape = (self.seq_len, self.emb_dim))

    angle = self.get_angle()

    pos[:, 0::2] = np.sin(angle[:, 0::2])
    pos[:, 1::2] = np.cos(angle[:, 1::2])
    pos = pos[np.newaxis, :]

    return tf.cast(pos, tf.float32)

  # [batch, seq_len, emb_dim]
  def call(self, input):
    # [seq_len, 1] * [1, emb_dim] -> [seq_len, emb_dim]
    pos_encoding = self.positional_encoding()

    return input + pos_encoding[:, :input.shape[1], :]

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, emb_dim, relative, max_len):
    super(MultiHeadAttention, self).__init__()
    self.num_heads   = num_heads
    self.d_k         = tf.cast(emb_dim // num_heads, tf.float32)
    self.dense_query = tf.keras.layers.Dense(emb_dim)
    self.dense_key   = tf.keras.layers.Dense(emb_dim)
    self.dense_value = tf.keras.layers.Dense(emb_dim)
    self.final       = tf.keras.layers.Dense(emb_dim)

    self.relative = relative
    self.max_len  = max_len

    if self.relative:
      # relative position can be shared across head
      self.relative_key = tf.keras.layers.Embedding(2*self.max_len + 1, emb_dim // num_heads)
      self.relative_val = tf.keras.layers.Embedding(2*self.max_len + 1, emb_dim // num_heads)

  def relative_weights(self, key, value):
    key_tmp    = tf.range(key.shape[2])
    dist_matrix = key_tmp[None,:] - key_tmp[:,None]
    dist_matrix = tf.clip_by_value(dist_matrix, clip_value_min = -self.max_len, clip_value_max = self.max_len)
    dist_matrix += self.max_len
    dist_matrix = tf.cast(dist_matrix, dtype = tf.int32)
    #print("[DEBUG] dist_matrix.shape: ", dist_matrix.shape)
    return self.relative_key(dist_matrix), self.relative_val(dist_matrix)

  def multi_heads(self, input):
    batch   = input.shape[0]
    seq_len = input.shape[1]
    input = tf.reshape(input, shape = (batch, seq_len, self.num_heads, self.d_k))
    return tf.transpose(input, perm = [0, 2, 1, 3])

  def call(self, query, key, value, mask):

    # (batch, seq_len, emb_dim) -> (batch, seq_len, emb_dim)
    query = self.dense_query(query)
    key   = self.dense_key(key)
    value = self.dense_value(value)

    # (batch, seq_len, emb_dim) -> (batch, num_heads, seq_len, emb_dim / num_heads)
    query    = self.multi_heads(query)
    key      = self.multi_heads(key)
    value    = self.multi_heads(value)

    # (batch, num_heads, seq_len, d_k) * (batch, num_heads, emb_dim, seq_len) -> (batch, num_heads, seq_len, seq_len)

    # [RELATIVE] key + relative_key
    # key += relative_key_weights (batch, 1, seq_len, d_k)
    scaled_qk = tf.matmul(query, key, transpose_b = True) / tf.math.sqrt(self.d_k)


    if self.relative:
      relative_key_weights, relative_val_weights = self.relative_weights(key, value)
      batch, num_heads, seq_len, d_k = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
      # [seq_len, batch*n_heads, d_k] * [seq_len, d_k, seq_len] -> [seq_len, batch*n_heads, seq_len]
      query_reshape = tf.reshape(tf.transpose(query, perm = [2, 0, 1, 3]), shape = (seq_len, batch*num_heads, d_k))
      relative_key  = tf.transpose(relative_key_weights, perm = [0, 2, 1])
      scaled_qrel = tf.matmul(query_reshape, relative_key) / tf.math.sqrt(self.d_k)
      # [seq_len, batch, n_heads, seq_len] -> [batch, n_heads, seq_len, seq_len]
      scaled_qrel = tf.transpose(tf.reshape(scaled_qrel, shape = (seq_len, batch, num_heads, seq_len)), perm = [1, 2, 0, 3])
      scaled_qk += scaled_qrel

    if mask is not None:
      scaled_qk += mask * (-1e9)

    # (이 부분 잘 이해가 안가는데 확인)
    # (batch, num_heads, seq_len, seq_len) -> (batch, num_heads, seq_len)
    weights = tf.nn.softmax(scaled_qk, axis = -1)

    # (batch, num_heads, seq_len) * (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, d_k)
    # [RELATIVE] val + relative_val (batch, 1, seq_len, d_k)
    # value += relative_val_weights
    outputs = tf.matmul(weights, value)

    if self.relative:
      # weights.shape [batch, num_heads, seq_len, seq_len], rel_key = [seq_len, seq_len, d_k]
      batch, num_heads, seq_len, d_k = value.shape[0], value.shape[1], value.shape[2], value.shape[3]
      weights_reshape = tf.reshape(tf.transpose(weights, perm = [2, 0, 1, 3]), shape = (seq_len, batch * num_heads, seq_len))
      relative_val = tf.matmul(weights_reshape, relative_val_weights)

      relative_val = tf.transpose(tf.reshape(relative_val, shape = (seq_len, batch, num_heads, d_k)), perm = [1, 2, 0, 3])
      outputs += relative_val

    # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
    outputs = tf.transpose(outputs, perm = [0, 2, 1, 3])
    # (batch, seq_len, num_heads, d_k) -> (batch, seq_len, emb_dim)
    outputs = tf.reshape(outputs, shape = (outputs.shape[0], outputs.shape[1], -1))
    outputs = self.final(outputs)
    return outputs

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, emb_dim, hidden_dim, relative, max_len):
    super(EncoderLayer, self).__init__()
    self.num_heads  = num_heads
    self.emb_dim    = emb_dim

    self.attention      = MultiHeadAttention(num_heads, emb_dim, relative, max_len)
    self.dropouts       = [tf.keras.layers.Dropout(0.1) for i in range(2)]
    self.layernorms     = [tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6) for i in range(2)]
    self.ffns           = [tf.keras.layers.Dense(hidden_dim, activation = 'relu'),
                           tf.keras.layers.Dense(emb_dim)]

  def call(self, inputs, pad_masks):

    outputs       = self.attention(query = inputs,
                                   key = inputs,
                                   value = inputs,
                                   mask  = pad_masks)
    #print("output.shape; ", outputs[0].shape, outputs[1].shape)
    outputs       = self.dropouts[0](outputs)
    first_outputs = self.layernorms[0](outputs + inputs)

    outputs = self.ffns[0](first_outputs)
    outputs = self.ffns[1](outputs)
    outputs = self.dropouts[1](outputs)
    outputs = self.layernorms[1](outputs + first_outputs)

    return outputs

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, emb_dim, hidden_dim, relative, max_len):
    super(DecoderLayer, self).__init__()
    self.num_heads = num_heads
    self.emb_dim   = emb_dim

    self.attentions      = [MultiHeadAttention(num_heads, emb_dim, relative, max_len), MultiHeadAttention(num_heads, emb_dim, 0, max_len)]
    self.dropouts        = [tf.keras.layers.Dropout(0.1) for i in range(3)]
    self.layernorms      = [tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-6) for i in range(3)]
    self.ffns            = [tf.keras.layers.Dense(hidden_dim, activation = 'relu'),
                            tf.keras.layers.Dense(emb_dim)]

  # Given input embeddings with positional encoding, generate output sequence embeddings.
  def call(self, dec_inputs, enc_outputs, look_ahead_masks, pad_masks):

    outputs          = self.attentions[0](query = dec_inputs,
                                          key = dec_inputs,
                                          value = dec_inputs,
                                          mask = look_ahead_masks)
    outputs          = self.dropouts[0](outputs)
    first_outputs    = self.layernorms[0](outputs + dec_inputs)

    outputs          = self.attentions[1](query = first_outputs,
                                          key = enc_outputs,
                                          value = enc_outputs,
                                          mask = pad_masks)
    outputs          = self.dropouts[1](outputs)
    second_outputs   = self.layernorms[1](outputs + first_outputs)

    outputs          = self.ffns[0](second_outputs)
    outputs          = self.ffns[1](outputs)
    outputs          = self.dropouts[2](outputs)
    final_outputs    = self.layernorms[2](outputs + second_outputs)

    return final_outputs

# positional_encoding class로 따로 빼서 relative, absolute인지에 따라 다르게 호출되는 방식으로 구현하기
# embedding scailing
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, num_heads, emb_dim, hidden_dim, vocab_size, relative, max_len):
    super(Transformer, self).__init__()

    self.num_layers = num_layers

    self.enc_padding_mask   = tf.keras.layers.Lambda(padding_mask, name='enc_padding_mask')
    self.dec_padding_mask    = tf.keras.layers.Lambda(padding_mask, name = 'dec_padding_mask')
    self.dec_look_ahead_mask = tf.keras.layers.Lambda(look_ahead_mask, name = 'dec_look_ahead_mask')

    self.enc_embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
    self.dec_embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)

    self.enc_pos  = PositionalEncoding(vocab_size, emb_dim)
    self.dec_pos  = PositionalEncoding(vocab_size, emb_dim)

    self.encoders = [EncoderLayer(num_heads, emb_dim, hidden_dim, relative, max_len)] + [EncoderLayer(num_heads, emb_dim, hidden_dim, 0, max_len) for i in range(num_layers-1)]
    self.decoders = [DecoderLayer(num_heads, emb_dim, hidden_dim, relative, max_len)] + [DecoderLayer(num_heads, emb_dim, hidden_dim, 0, max_len) for i in range(num_layers-1)]

    self.final_layer = tf.keras.layers.Dense(vocab_size)
    self.relative    = relative

  def call(self, enc_input, dec_input):

    enc_padding = self.enc_padding_mask(enc_input)
    dec_padding = self.dec_padding_mask(enc_input)
    enc_input   = self.enc_embedding(enc_input)

    if not self.relative:
      enc_input   = self.enc_pos(enc_input)

    for i in range(self.num_layers):
      enc_input = self.encoders[i] (enc_input, enc_padding)

    dec_look_ahead_masks = self.dec_look_ahead_mask(dec_input)
    dec_output  = self.dec_embedding(dec_input)

    if not self.relative:
      dec_output  = self.dec_pos(dec_output)

    for i in range(self.num_layers):
      dec_output = self.decoders[i](dec_output, enc_input, dec_look_ahead_masks, dec_padding)

    final_output = self.final_layer(dec_output)

    return final_output
