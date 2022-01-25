import tensorflow as tf
import argparse
import re

from model import *
from load_data import *
from utils import *

'''
reference: https://wikidocs.net/31379

'''

def evaluate(sentence, model, tokenizer):
    MAX_LENGTH = 40
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)
    print(START_TOKEN, END_TOKEN)
    for i in range(MAX_LENGTH):
        predictions = model(enc_input = sentence, dec_input = output, training = False)

        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

    output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence, model, tokenizer):
    prediction = evaluate(sentence, model, tokenizer)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()

    return sentence

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
    small_transformer = Transformer(num_layers = args.num_layer,
                                    num_heads = args.num_heads,
                                    emb_dim = args.emb_dim,
                                    hidden_dim = args.hidden_dim,
                                    vocab_size = 9000,
                                    relative = args.is_relative,
                                    max_len = args.max_len
                                    )

    model_type = 'relative' if args.is_relative == 1 else 'absolute'
    learning_rate = CustomSchedule(128)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    small_transformer.compile(optimizer = optimizer, loss = loss_function, metrics = [accuracy])

    small_transformer.load_weights(f"{args.save_path}/{model_type}_model.tf")
    _, tokenizer = load_data_korean()

    input_sentence = "커피 좋아하세요?"
    predict(input_sentence, small_transformer, tokenizer)

    input_sentence = "고민이 있어"
    predict(input_sentence, small_transformer, tokenizer)
