import os
import joblib
import argparse
import logging
import numpy as np
import tensorflow as tf
from src.utils.common import read_yaml, create_directories
from src.utils.models import build_model, generate_text
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 4" 

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
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    checkpoint_dir = os.path.join(model_dir_path, artifacts["CHECKPOINT_DIR"])

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    vocab_data_file = os.path.join(featurized_data_dir_path, artifacts["VOCAB_DATA"])
    text_as_int_file = os.path.join(featurized_data_dir_path, artifacts["TEXT_AS_INTEGER"])

    embedding_dim = params["train"]["embedding_dim"]
    rnn_units = params["train"]["rnn_units"]
    start_string = params["evaluate"]["start_string"]
    no_of_chars_to_gen = params["evaluate"]["no_of_chars_to_gen"]

    vocab = joblib.load(vocab_data_file)
    text_as_int = joblib.load(text_as_int_file)
    vocab_len = len(vocab)
    print(f"Vocab length: {vocab_len}")
    char2idx = {uniChar: idx for idx, uniChar in enumerate(vocab)}
    idx2char_DICT = {val: key for key, val in char2idx.items()}
    idx2char = np.array(vocab)
    
    # Restoring checkoint - 
    tf.train.latest_checkpoint(checkpoint_dir)

    model_from_ckpt = build_model(
    vocab_size = vocab_len,
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
    batch_size = 1 
    )

    print(model_from_ckpt.summary())

    model_from_ckpt.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model_from_ckpt.build(tf.TensorShape([1, None]))

    print(model_from_ckpt.summary())

    result = generate_text(model=model_from_ckpt, start_string=start_string,
                            char2idx=char2idx, idx2char=idx2char,
                            no_of_chars_to_gen=no_of_chars_to_gen)
    print(result)



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
