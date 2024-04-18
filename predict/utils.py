import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load

def get_file_name(feature_file: str):
    return os.path.splitext(os.path.basename(feature_file))[0]


def get_X_and_y_v0_density(feature_file: str):
    """get features X and truth y. Use density nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
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

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y

def get_X_and_y_v1_value(feature_file: str):
    """get features X and truth y. Use value nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_per_row',
                'min_nnz_per_row',
                'max_nnz_per_row',
                'std_dev_nnz_per_row']
    X = df[features].to_numpy()

    speedup = df[['speed_to_naive']].to_numpy()
    y = [ 1 if x >= 1.1 else 0 for x in speedup ]

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y

def evaluate(truth,
             predict):
    stat = {
        "accuracy": [accuracy_score(y_true=truth, y_pred=predict)],
        "precision": [precision_score(y_true=truth, y_pred=predict)],
        "recall": [recall_score(y_true=truth, y_pred=predict)],
        "f1": [f1_score(y_true=truth, y_pred=predict)]
    }

    return stat


def save_ref_perf(ref_perf: dict,
                  feature_file: str,
                  output_dir: str,
                  desc: str):
    basename = get_file_name(feature_file)
    ref_file = os.path.join(output_dir, F"{basename}.{desc}.csv")
    df = pd.DataFrame(data=ref_perf)
    df.to_csv(ref_file, index=False)
    print(df)


def save_model(classifier,
               feature_file: str,
               output_dir: str,
               desc: str):
    basename = get_file_name(feature_file)
    model_file = os.path.join(output_dir, F"{basename}.{desc}.joblib")
    dump(classifier, model_file)
    print(F"Saved model {classifier} to {model_file} .")


def save_train_perf(train_perf: dict,
                    feature_file: str,
                    output_dir: str,
                    desc: str):
    basename = get_file_name(feature_file)
    train_file = os.path.join(output_dir, F"{basename}.{desc}.csv")
    df = pd.DataFrame(data=train_perf)
    df.to_csv(train_file, index=False)
    print(df)