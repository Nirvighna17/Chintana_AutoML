import pandas as pd
import numpy as np


def full_transformation_pipeline(
    df,
    scale_method="standard",
    include_polynomial=False,
    apply_log=True,
    smart_encode=True,
    drop_corr=True,
    drop_low_var=True,
    extract_datetime=True
):
    # Drop duplicate columns (if any)
    df = df.loc[:, ~df.columns.duplicated()]

    # 1. Extract datetime features
    if extract_datetime:
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        for col in datetime_cols:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
        df.drop(columns=datetime_cols, inplace=True)

    # 2. Apply log transformation to skewed features
    if apply_log:
        for col in df.select_dtypes(include=['float64', 'int64']):
            if (df[col] > 0).all() and abs(df[col].skew()) > 1:
                df[col] = df[col].apply(lambda x: np.log1p(x))

    # 3. Encode categorical features
    if smart_encode:
        for col in df.select_dtypes(include='object').columns:
            if df[col].nunique() < 10:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            else:
                df.drop(columns=col, inplace=True)

    # 4. Drop highly correlated features
    if drop_corr:
        corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        df.drop(columns=to_drop, inplace=True)

    # 5. Remove low variance features
    if drop_low_var:
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        transformed = selector.fit_transform(df[num_cols])
        df = pd.DataFrame(transformed, columns=num_cols[selector.get_support(indices=True)])

    # 6. Scaling
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'maxabs': MaxAbsScaler()
    }
    scaler = scaler_dict.get(scale_method, StandardScaler())
    df[df.columns] = scaler.fit_transform(df)

    # 7. Polynomial Features
    if include_polynomial:
        from sklearn.preprocessing import PolynomialFeatures
        pf = PolynomialFeatures(degree=2, include_bias=False)
        df_poly = pf.fit_transform(df)
        df = pd.DataFrame(df_poly)

    return df
