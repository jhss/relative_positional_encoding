import tensorflow as tf
import numpy as np

# [JH] Clear
# Given a token sequence, convert it into a padding mask.
# [batch, seq_len] -> [batch, 1, 1, seq_len]
def padding_mask(inputs):
  masks = tf.cast(tf.equal(inputs, 0), tf.float32)
  return masks[:, tf.newaxis, tf.newaxis, :]

def look_ahead_mask(inputs):
  seq_len = inputs.shape[1]
  ret = np.zeros(shape=(seq_len, seq_len))
  ret[np.tril_indices(seq_len)] = 1
  look_ahead = tf.cast(1 - ret, tf.float32)
  padding    = padding_mask(inputs)

  return tf.maximum(look_ahead, padding)

def loss_function(y_true, y_pred):
  MAX_LENGTH = 40
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH -1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
