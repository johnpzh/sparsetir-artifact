import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score


def RSE(mse: float,
        n: int):
    """Calculate the Residual Standard Error (RSE)

    Args:
        mse (float): Mean Squared Error (MSE)
        n (int): number of elements

    Returns:
        float: RSE
    """
    p = 1 # assuming a simple linear regression model with one independent variable
    df = n - p - 1
    rse = np.sqrt(mse / df)
    return rse


def evaluate(table: dict):
    truth_num_parts = table["truth_num_partitions"]
    predict_num_parts = table["predict_num_partitions"]
    truth_max_bucket_width = table["truth_max_bucket_width"]
    predict_max_bucket_width = table["predict_max_bucket_width"]

    # Mean Squared Error (MSE)
    mse_num_parts = mean_squared_error(truth_num_parts, predict_num_parts)
    mse_max_bucket_width = mean_squared_error(truth_max_bucket_width, predict_max_bucket_width)

    # Root Mean Squared Error (RMSE)
    rmse_num_parts = np.sqrt(mse_num_parts)
    rmse_max_bucket_width = np.sqrt(mse_max_bucket_width)

    # Residual Standard Error (RSE)
    rse_num_parts = RSE(mse_num_parts, len(truth_num_parts))
    rse_max_bucket_width = RSE(mse_max_bucket_width, len(truth_max_bucket_width))

    # Coefficient of Determination or R-Squared (R2)
    r2_num_parts = r2_score(truth_num_parts, predict_num_parts)
    r2_max_bucket_width = r2_score(truth_max_bucket_width, predict_max_bucket_width)

    columns = {
        "Parameters": ["num_partitions", "max_bucket_width"],
        "Root_Mean_Squared_Error_(RMSE)": [rmse_num_parts, rmse_max_bucket_width],
        "Residual_Standard_Error_(RSE)": [rse_num_parts, rse_max_bucket_width],
        "R-Squared_(R2)": [r2_num_parts, r2_max_bucket_width]
    }

    df = pd.DataFrame(data=columns)
    print(df)

    return df


def save_to_csv(table: dict,
                basename: str):
    # Pandas settings
    pd.set_option("display.width", 800)
    # pd.set_option("display.max_columns", None)

    output_file = F"{basename}.linear_regression.predict_output.csv"
    df = pd.DataFrame(data=table)
    print(df)
    df.to_csv(output_file, index=False)

    # Evaluate prediction results
    eval_df = evaluate(table)
    eval_df.to_csv(output_file, index=False, mode='a')


def predict(feature_file: str,
            trained_coef_file: str):
    output_dir = "output"
    df_feature = pd.read_csv(feature_file)
    df_trained_b = pd.read_csv(trained_coef_file)

    truth_y_num_parts = df_feature['best_num_partitions'].to_numpy()
    truth_y_max_bucket_width = df_feature['best_max_bucket_width'].to_numpy()
    X = df_feature.drop(columns=['name', 'best_num_partitions', 'best_max_bucket_width', 'exe_time_hyb', 'exe_time_naive', 'speed_to_naive']).to_numpy()
    b_num_parts = df_trained_b['b_num_partitions'].to_numpy()
    b_max_bucket_width = df_trained_b['b_max_bucket_width'].to_numpy()

    # Predition
    yhat_num_parts = X.dot(b_num_parts)
    yhat_max_bucket_width = X.dot(b_max_bucket_width)

    # Save to csv
    basename = os.path.join(output_dir, F"{os.path.splitext(os.path.basename(feature_file))[0]}")
    table = {
        "truth_num_partitions": truth_y_num_parts,
        "predict_num_partitions": yhat_num_parts,
        "truth_max_bucket_width": truth_y_max_bucket_width,
        "predict_max_bucket_width": yhat_max_bucket_width
    }
    save_to_csv(table=table,
                basename=basename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parse searching configurations")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    parser.add_argument("--trained-coefficients", "-c", type=str, help="coefficient csv file from training")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    trained_coef_file = args.trained_coefficients
    predict(feature_file,
            trained_coef_file)