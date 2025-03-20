#!/usr/bin/env python
# coding: utf-8

def main(df):

    
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from autoviz.AutoViz_Class import AutoViz_Class
    from sklearn.feature_selection import mutual_info_classif


    # Load your dataset
    df = df

    # 1. Remove columns with complete null values or columns with the same values throughout
    def remove_null_constant_columns(df):
        # Remove columns with all null values
        df = df.dropna(axis=1, how='all')
        # Remove columns with the same value throughout
        df = df.loc[:, df.nunique() != 1]
        return df

    df = remove_null_constant_columns(df)

    # 2. Separate numerical and categorical columns into separate dataframes
    numerical_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    # 3. Replace missing values using median for numerical and mode for categorical columns
    def impute_missing_values(df, strategy, column_type):
        if column_type == 'numerical':
            imputer = SimpleImputer(strategy=strategy)
        else:
            imputer = SimpleImputer(strategy='most_frequent')
        
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_imputed

    numerical_df = impute_missing_values(numerical_df, strategy='median', column_type='numerical')
    categorical_df = impute_missing_values(categorical_df, strategy='most_frequent', column_type='categorical')

    # Combine the numerical and categorical dataframes back into one dataframe
    df_cleaned = pd.concat([numerical_df, categorical_df], axis=1)
    
    return df_cleaned
