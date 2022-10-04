
import os
import shutil
import random
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from src.utils.common import read_yaml, create_directories


STAGE = "STAGE_1"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
   ## Converting XML data to csv
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    source_data = config["source_data"]
    input_data = os.path.join(source_data["data_dir"], source_data["data_file"])
    sheet_name = source_data["sheet_name"]

    df=pd.read_excel(input_data, sheet_name=sheet_name)
    print(df.head())


    # split = params["prepare"]["split"]
    # seed = params["prepare"]["seed"]
    # random.seed(seed)

    # artifacts = config["artifacts"]
    # prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    # create_directories([prepared_data_dir_path])

    # train_data_path = os.path.join(prepared_data_dir_path, artifacts["TRAIN_DATA"])
    # test_data_path = os.path.join(prepared_data_dir_path, artifacts["TEST_DATA"])
    


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
