import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
import time
import utils


def predict(feature_file: str,
            model_file: str):
    output_dir = "output"

    # Predict
    X, y_truth = utils.get_X_and_y_v0_density(feature_file)
    clf = load(model_file)
    start_time = time.perf_counter()
    predict = clf.predict(X)
    end_time = time.perf_counter()

    # Evaluate
    stat = utils.evaluate(truth=y_truth, predict=predict)
    stat["ref_time(s)"] = [end_time - start_time]

    # Save 
    utils.save_ref_perf(stat,
                        feature_file,
                        output_dir,
                        desc="RandomForestClassifier.ref_perf.density")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test on dataset")
    parser.add_argument("--features", "-f", type=str, help="features of test cases in a csv file")
    parser.add_argument("--model", "-m", type=str, help="model to use")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    model_file = args.model

    predict(feature_file, model_file)