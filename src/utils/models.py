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

def generate_text(model, start_string, char2idx, idx2char, no_of_chars_to_gen=1000):
    # convert the input text to nos.
    input_val = [char2idx[s] for s in start_string] # text converted to int
    input_val = tf.expand_dims(input_val, 0) # [] ->> [1, ]

    text_generated = list()

    temperature = 1.0


    model.reset_states() # Resetting the previous states if any while predictions.
    for i in range(no_of_chars_to_gen):
        predictions = model(input_val)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        # print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy() # It will draw some sample out of it
        # print(predicted_id)
        
        input_val = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)