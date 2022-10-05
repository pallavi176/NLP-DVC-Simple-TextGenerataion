import tensorflow as tf

def build_model(
    vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
          tf.keras.layers.GRU(rnn_units,
                    return_sequences=True,                  # True: seq2seq, False: seq2vec
                    stateful=True,
                    recurrent_initializer='glorot_uniform'),
          tf.keras.layers.Dense(vocab_size)
    ])
    return model

def model_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)  #logits: raw output from dense neurons without any activation function
