import os
import argparse
import logging
import joblib
import numpy as np
import tensorflow as tf
from src.utils.common import read_yaml, create_directories
from src.utils.data_process import split_input_target
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 2" 

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
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepared_data_dir_path, artifacts["TRAIN_DATA"])

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    create_directories([featurized_data_dir_path])

    vocab_data_file = os.path.join(featurized_data_dir_path, artifacts["VOCAB_DATA"])
    text_as_int_file = os.path.join(featurized_data_dir_path, artifacts["TEXT_AS_INTEGER"])

    featurized_dataset_dir = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATASET_DIR"])
    #create_directories([featurized_dataset_dir])
    #featurized_dataset_file_path = os.path.join(featurized_dataset_dir, artifacts["FEATURIZED_DATASET_FILE"])

    seq_length = params["featurize"]["seq_length"]
    batch_size = params["featurize"]["batch_size"]
    buffer_size = params["featurize"]["buffer_size"]

    fb = open(train_data_path,"r") 
    text_data = fb.read()
    fb.close()

    # Get unique characters or letters that we have
    vocab = sorted(set(text_data))  #vocab: list of all the unique characters
    joblib.dump(vocab, vocab_data_file)
    logging.info(f"Vocab length: {len(vocab)}")

    char2idx = {uniChar: idx for idx, uniChar in enumerate(vocab)}
    idx2char_DICT = {val: key for key, val in char2idx.items()}
    idx2char = np.array(vocab)

    # all text is now represented as integer
    text_as_int = np.array([char2idx[c] for c in text_data])
    joblib.dump(text_as_int, text_as_int_file)

    # We have to tell the model that what will be the maximum length that we are going to process at every sequence
    examples_per_epoch = len(text_data)//(seq_length + 1) 

    #Create training examples per target
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)  # create tensorflow dataset

    # creating batches of 101 & dropping the remainder
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True) 

    # First Citize(Input) -> irst Citizen(Target)
    dataset = sequences.map(split_input_target)

    # Create Training batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    #dataset.save(featurized_dataset_file_path)
    tf.saved_model.save(dataset, featurized_dataset_dir)
    #joblib.dump(dataset, featurized_dataset_file_path)

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
