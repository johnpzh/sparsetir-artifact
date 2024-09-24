import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import argparse
import sys
import os
import math
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_validate
# from joblib import dump, load
import time
import utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "output"
MODEL_DIR = "models"


def train_and_infer(train_file: str,
                    infer_file: str):
    output_dir = OUTPUT_DIR

    # Get features X and truth y
    X_train, y_train = utils.get_X_value_and_y_num_parts(train_file)
    X_infer, y_truth = utils.get_X_value_and_y_num_parts(infer_file)

    print(f"X_train: {len(X_train)} y_train: {len(y_train)}")
    table = {
        "data_size": [],
        "predict_accuracy": []
    }

    # size = 1
    for percent in range(1, 101):
    # while size < len(X_train):
        percent /= 100
        size = int(len(X_train) * percent)
        # percent = size / len(X_train)
        idx = np.random.permutation(len(X_train))[:size]
        # print(f"idx: {idx}")
        X_train_part = X_train[idx]
        y_train_part = y_train[idx]
        
        # Model
        est_name = "RandomForest"
        clf = utils.estimators[est_name]
        clf = make_pipeline(StandardScaler(), clf)

        # Train
        clf.fit(X_train_part, y_train_part)

        # Predict
        stat = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        y_predict = clf.predict(X_infer)
        utils.evaluate_into_stat(truth=y_truth, predict=y_predict, stat=stat)

        # table["data_size"].append(percent)
        table["data_size"].append(size)
        table["predict_accuracy"].append(stat["accuracy"][0])
        print(f"percent: {percent} size: {size} accuracy: {stat['accuracy'][0]}")
        
        # size *= 2

    # Save 
    basename = utils.get_file_name(train_file)
    output_file = os.path.join(output_dir, F"{basename}.predict_results.trend.csv")
    df = pd.DataFrame(data=table)
    df.to_csv(output_file, index=False)
    print(df.to_string())
    


# def train_and_infer(train_file: str,
#                     infer_file: str):
#     output_dir = OUTPUT_DIR

#     # Get features X and truth y
#     X_train, y_train = utils.get_X_value_and_y_num_parts(train_file)
#     X_infer, y_truth = utils.get_X_value_and_y_num_parts(infer_file)

#     print(f"X_train: {len(X_train)} y_train: {len(y_train)}")
#     table = {
#         "data_size": [],
#         "predict_accuracy": []
#     }

#     size = 1
#     # for percent in range(1, 101):
#     bound = len(X_train)
#     while size < bound:
#         # percent /= 100
#         # size = int(bound * percent)
#         percent = size / bound
#         idx = np.random.permutation(bound)[:size]
#         # print(f"idx: {idx}")
#         X_train_part = X_train[idx]
#         y_train_part = y_train[idx]
        
#         # Model
#         est_name = "RandomForest"
#         clf = utils.estimators[est_name]
#         clf = make_pipeline(StandardScaler(), clf)

#         # Train
#         clf.fit(X_train_part, y_train_part)

#         # Predict
#         stat = {
#             "accuracy": [],
#             "precision": [],
#             "recall": [],
#             "f1": []
#         }
#         y_predict = clf.predict(X_infer)
#         utils.evaluate_into_stat(truth=y_truth, predict=y_predict, stat=stat)

#         # table["data_size"].append(percent)
#         table["data_size"].append(size)
#         table["predict_accuracy"].append(stat["accuracy"][0])
#         print(f"percent: {percent} size: {size} accuracy: {stat['accuracy'][0]}")
        
#         size *= 2
    
#     size = bound
#     percent = size / bound
#     idx = np.random.permutation(bound)[:size]
#     # print(f"idx: {idx}")
#     X_train_part = X_train[idx]
#     y_train_part = y_train[idx]
    
#     # Model
#     est_name = "RandomForest"
#     clf = utils.estimators[est_name]
#     clf = make_pipeline(StandardScaler(), clf)

#     # Train
#     clf.fit(X_train_part, y_train_part)

#     # Predict
#     stat = {
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": []
#     }
#     y_predict = clf.predict(X_infer)
#     utils.evaluate_into_stat(truth=y_truth, predict=y_predict, stat=stat)

#     # table["data_size"].append(percent)
#     table["data_size"].append(size)
#     table["predict_accuracy"].append(stat["accuracy"][0])
#     print(f"percent: {percent} size: {size} accuracy: {stat['accuracy'][0]}")

#     # Save 
#     basename = utils.get_file_name(train_file)
#     output_file = os.path.join(output_dir, F"{basename}.predict_results.trend.csv")
#     df = pd.DataFrame(data=table)
#     df.to_csv(output_file, index=False)
#     print(df.to_string())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train models for number of partitions")
    parser.add_argument("--train-file", "-t", type=str, help="train csv file")
    parser.add_argument("--infer-file", "-i", type=str, help="test csv file for prediction")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    train_file = args.train_file
    infer_file = args.infer_file

    train_and_infer(train_file, infer_file)
