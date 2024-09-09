# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import argparse
import sys
import os
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_validate
# from joblib import dump, load
import time
import utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "output"
MODEL_DIR = "models"

def train_and_infer(train_file: str,
                    infer_file: str):
    output_dir = OUTPUT_DIR

    # Get features X and truth y
    X_train, y_train = utils.get_X_value_and_y_num_parts(train_file)
    X_infer, y_truth = utils.get_X_value_and_y_num_parts(infer_file)

    stat = {
        "name": [],
        "train_time(s)": [],
        "infer_time(s)": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for est_name, clf in utils.estimators.items():
    # for est_name, est_fun in utils.estimators.items():
        print(F"Use estimator {est_name}")
        # Iterate all estimators
        stat["name"].append(est_name)
        clf = make_pipeline(StandardScaler(), clf)

        # Train
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        end_time = time.perf_counter()
        stat["train_time(s)"].append(end_time - start_time)

        # Predict
        start_time = time.perf_counter()
        y_predict = clf.predict(X_infer)
        end_time = time.perf_counter()
        stat["infer_time(s)"].append(end_time - start_time)
        utils.evaluate_into_stat(truth=y_truth, predict=y_predict, stat=stat)

        # Save model
        utils.save_model(classifier=clf,
                         feature_file=train_file,
                         output_dir=MODEL_DIR,
                         desc=est_name)


    # Save 
    basename = utils.get_file_name(train_file)
    output_file = os.path.join(output_dir, F"{basename}.predict_results.csv")
    df = pd.DataFrame(data=stat)
    df.to_csv(output_file, index=False)
    print(df.to_string())
    

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
