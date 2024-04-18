import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from joblib import dump, load
import time
import utils


def train(feature_file: str):
    output_dir = "output"

    # Get features X and truth y
    X, y = utils.get_X_and_y_v1_value(feature_file)

    # Fit
    clf = RandomForestClassifier(random_state=0)
    start_time = time.perf_counter()
    clf.fit(X, y)
    end_time = time.perf_counter()

    stat = {
        "train_time(s)": [end_time - start_time]
    }

    # Save the fitted model to file?
    utils.save_model(clf,
                     feature_file,
                     output_dir,
                     desc="RandomForestClassifier.abs_value")
    
    utils.save_train_perf(stat,
                          feature_file,
                          output_dir,
                          desc="RandomForestClassifier.train_perf.abs_value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train on dataset")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features

    train(feature_file)
