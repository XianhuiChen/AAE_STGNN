import pandas as pd
import parse
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def read_county_data(level = 'county'):
    if level == 'county':
        county = 'County'
        df = pd.read_csv(parse.args.county_file)
        sdi = pd.read_csv(parse.args.sdi_county_file)
    else:
        county = 'ZCTA'
        df = pd.read_csv(parse.args.zcta_file)
        sdi = pd.read_csv(parse.args.sdi_zcta_file)
        df = df[df['ZCTA_population'] > 500]
    df.fillna(0, inplace=True)
    df = df.sort_values(by="tm")
    df['label_count'] = df['opioid.overdose']
    df['label.per'] = df['opioid.overdose'] / df[f'{county}_population']
    df['population'] = df[f'{county}_population']
    for i in range(2, 13):
        for col in parse.args.feature_cols:
            if 'label' in col:
                continue
            df[f'{col}_rolling_{i}'] = df.groupby(county)[col].rolling(window=i).sum().reset_index(level=0, drop=True).fillna(0)
            df[f'{col}.pos_rolling_{i}'] = df.groupby(county)[col + '.pos'].rolling(window=i).sum().reset_index(level=0, drop=True).fillna(0)
            df[f'{col}.per_rolling_{i}'] = df[f'{col}.pos_rolling_{i}'] / (0.01 + df[f'{col}_rolling_{i}'].fillna(0))

        col = 'label_count'
        df[f'{col}_rolling_{i}'] = df.groupby(county)[col].rolling(window=i).mean().reset_index(level=0, drop=True).fillna(0)
        df[f'label.per_rolling_{i}'] = df[f'{col}_rolling_{i}'] / df[f'{county}_population']
        df[f'label_rolling_{i}'] = df.groupby(county)[f"label_count_rolling_{i}"].shift(- parse.args.prediction_window)
        

    df['label'] = df.groupby(county)["label_count"].shift(- parse.args.prediction_window)

    df.fillna(0, inplace=True)
    return df, sdi


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    Parameters:
        y_true: array-like, true values
        y_pred: array-like, predicted values

    Returns:
        SMAPE value in percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)

    # Avoid division by zero
    smape = np.where(denominator == 0, 0, diff / denominator)

    return np.mean(smape) * 100

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)

    Parameters:
        y_true: array-like, true values
        y_pred: array-like, predicted values

    Returns:
        MAE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    diff = np.abs(y_pred - y_true)
    mae = np.mean(diff)

    return mae
def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)

    Parameters:
        y_true: array-like, true values
        y_pred: array-like, predicted values

    Returns:
        RMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    return rmse
