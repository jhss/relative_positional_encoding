import sys
import argparse

import torch
import torch.optim as optim
from tqdm import tqdm

from model import *
from load_data import *
from utils import *

def train(args):
    dataset, _ = load_data_korean()

    small_transformer = Transformer(num_layers = args.num_layer,
                                    num_heads = args.num_heads,
                                    emb_dim = args.emb_dim,
                                    hidden_dim = args.hidden_dim,
                                    vocab_size = 9000,
                                    relative = args.is_relative,
                                    max_len = args.max_len
                                    )

    tf.keras.backend.clear_session()

    learning_rate = CustomSchedule(128)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    small_transformer.compile(optimizer = optimizer, loss = loss_function, metrics=[accuracy])

    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
      for step, data in tqdm(enumerate(dataset)):
        enc_inputs, dec_inputs, outputs = data[0]['inputs'], data[0]['dec_inputs'], data[1]['outputs']

        with tf.GradientTape() as tape:
          logits = small_transformer(enc_inputs, dec_inputs)
          loss   = loss_function(outputs, logits)

        #print("logits: ", tf.math.argmax(logits, axis = 2))

        grads = tape.gradient(loss, small_transformer.trainable_variables)
        optimizer.apply_gradients(zip(grads, small_transformer.trainable_variables))

      print(f"epoch: {epoch}, loss: {loss}")

    model_type = 'relative' if args.is_relative == 1 else 'absolute'
    small_transformer.save_weights(f"{args.save_path}/{model_type}_model.tf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_layer', type = int, default = 2, help = 'the number of layers in an encoder and decoder')
    parser.add_argument('--emb_dim', type = int, default = 256, help = 'embedding dimmension of the representation')
    parser.add_argument('--num_heads', type = int, default = 8, help = 'the number of multi-heads in an encoder and decoder')
    parser.add_argument('--hidden_dim', type = int, default = 512, help = 'hidden dimension of a feed forward netowrk')
    parser.add_argument('--is_relative', type = int, default = 0, help = 'train a transformer with absolute or relative positional encoding')
    parser.add_argument('--max_len', type = int, default = 40, help = 'maximum length of a sentence')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch_size')
    parser.add_argument('--num_epochs', type = int, default = 50, help = 'num epochs')
    parser.add_argument('--save_path', type = str, default = './model_weights', help = 'directory for a trained model')

    args = parser.parse_args()
    print("num_layers: ", args.num_layer, " hidden:dim: ", args.hidden_dim, " emb_dim: ", args.emb_dim)
    train(args)
