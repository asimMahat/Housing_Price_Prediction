import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_columns(csv_path="train.csv", test=False):
    """
    Return cleaned df, numerical_column in x,  numerical_column in y,  and categorical columns
    """
    drop_columns = ["MoSold", "YrSold", "Id"]

    train_df = pd.read_csv(csv_path)
    missing_values = (train_df.isnull().sum().sort_values(ascending=False))
    train_df = train_df.drop((missing_values[missing_values > 1]).index,1)
    train_df = train_df.dropna()
    train_df = train_df.drop(drop_columns, axis=1)
    num_col_x = train_df.select_dtypes(exclude=['object']).columns.tolist()
    cat_col = train_df.select_dtypes(include=['object']).columns.tolist() # Object indicates a column has text

    if not test:
        num_col_x.remove('SalePrice')
    num_col_y = ["SalePrice"]
    return train_df, num_col_x, num_col_y, cat_col


def get_scaled_train_df(train_df, num_col_x, num_col_y):
    """
    Return the scaled df and the scaler used for scaling
    """

    standard_scaler_x = StandardScaler()
    scaled_num_df_x = standard_scaler_x.fit_transform(train_df[num_col_x])
    standard_scaler_y = StandardScaler()
    scaled_num_df_y = standard_scaler_y.fit_transform(train_df[num_col_y])

    train_df[num_col_x] = scaled_num_df_x
    train_df[num_col_y] = scaled_num_df_y

    return train_df, standard_scaler_x, standard_scaler_y
