import os
import joblib
import argparse
import logging
import numpy as np
import tensorflow as tf
from src.utils.common import read_yaml, create_directories
from src.utils.models import build_model, model_loss
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 3" 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    vocab_data_file = os.path.join(featurized_data_dir_path, artifacts["VOCAB_DATA"])
    text_as_int_file = os.path.join(featurized_data_dir_path, artifacts["TEXT_AS_INTEGER"])
    featurized_dataset_dir = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATASET_DIR"])
    #featurized_dataset_file_path = os.path.join(featurized_dataset_dir, artifacts["FEATURIZED_DATASET_FILE"])

    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    create_directories([model_dir_path])
    checkpoint_path = os.path.join(model_dir_path, artifacts["CHECKPOINT_DIR"])

    batch_size = params["featurize"]["batch_size"]
    embedding_dim = params["train"]["embedding_dim"]
    rnn_units = params["train"]["rnn_units"]
    optimizer = params["train"]["optimizer"]
    epochs = params["train"]["epochs"]
    verbose = params["train"]["verbose"]

    vocab = joblib.load(vocab_data_file)
    text_as_int = joblib.load(text_as_int_file)
    dataset = tf.saved_model.load(featurized_dataset_dir)

    vocab_len = len(vocab)
    print(f"Vocab length: {vocab_len}")

    char2idx = {uniChar: idx for idx, uniChar in enumerate(vocab)}
    idx2char_DICT = {val: key for key, val in char2idx.items()}
    idx2char = np.array(vocab)

    model = build_model(
    vocab_size = vocab_len,
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
    batch_size = batch_size
    )

    print(model.summary())
    logging.info(f"Model Summary: \n{model.summary()}")

    model.compile(optimizer=optimizer, loss=model_loss)

    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_prefix,
        save_weights_only = True
    )

    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback], verbose=verbose)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
