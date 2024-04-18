import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from joblib import dump, load

def save_to_csv(table: dict,
                basename: str):
    output_file = F"{basename}.speedup_gt_130.csv"
    df = pd.DataFrame(data=table)
    print(df)
    df.to_csv(output_file, index=False)


def get_file_name(feature_file: str):
    return os.path.splitext(os.path.basename(feature_file))[0]


def get_X_and_y(feature_file: str):
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_density_per_row',
                'min_nnz_density_per_row',
                'max_nnz_density_per_row',
                'std_dev_nnz_density_per_row']
    X = df[features].to_numpy()

    speedup = df[['speed_to_naive']].to_numpy()
    y = [ 1 if x >= 1.1 else 0 for x in speedup ]

    # test
    print(F"X: {X}")
    print(F"y: {y}")
    # end test

    return X, y


def main(feature_file: str):
    output_dir = "output"

    # Get features X and truth y
    X, y = get_X_and_y(feature_file)

    # Fit
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    # Save the fitted model to file?
    filename = get_file_name(feature_file)
    model_file = F"{filename}.RandomForestClassifier.joblib"
    dump(clf, model_file)

    # Predict
    clf = load(model_file)
    predict = clf.predict(X)
    print(F"predict: {predict}")

    # Evaluate
    result = cross_validate(clf, X, predict)
    print(F"result['test_score']: {result['test_score']}")

    # Try other features?

    # Try Auto-tuner?

    # Try different model?

    #

    #######

    # # Select speedup >= 1.30x
    # # above_130 = df[df["speed_to_naive"] >= 2.0]

    # # Drop duplicated matrices by names
    # # above_130.drop_duplicates(subset=['name'], inplace=True)
    # refined = df.drop_duplicates(subset=['name']).drop(columns=['K'])

    # basename = os.path.join(output_dir, F"{os.path.splitext(os.path.basename(feature_file))[0]}")
    # output_file = F"{basename}.binary_decision_train.csv"
    # print(refined)
    # refined.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Decide if hyb to be used")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    main(feature_file)