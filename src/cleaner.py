import pandas as pd
def clean_data(
    df,
    drop_duplicates=True,
    drop_high_null_columns=True,
    handle_missing_values=True,
    fix_column_types=True,
    strip_whitespace=True,
    remove_constant_columns=True,
    remove_outliers_iqr=True
):
    if drop_duplicates:
        df = df.drop_duplicates()

    if drop_high_null_columns:
        null_thresh = 0.5
        df = df.dropna(thresh=int((1 - null_thresh) * len(df)), axis=1)

    if handle_missing_values:
        skew_threshold = 1
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    skew = df[col].skew()
                    if abs(skew) < skew_threshold:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    if fix_column_types:
        for col in df.select_dtypes(include='object').columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    if strip_whitespace:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

    if remove_constant_columns:
        df = df.loc[:, df.nunique() > 1]

    if remove_outliers_iqr:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
