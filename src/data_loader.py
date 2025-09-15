import pandas as pd
import seaborn as sns
from sklearn.datasets import (
    load_iris, load_wine, load_diabetes, load_breast_cancer,
    fetch_california_housing, fetch_covtype
)


def get_predefined_dataset(name):
    if name == "Iris":
        return load_iris(as_frame=True).frame
    elif name == "Wine":
        return load_wine(as_frame=True).frame
    elif name == "Diabetes":
        return load_diabetes(as_frame=True).frame
    elif name == "Breast Cancer":
        return load_breast_cancer(as_frame=True).frame
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df['target'] = data.target
        return df
    elif name == "Covertype":
        data = fetch_covtype(as_frame=True)
        df = data.frame
        df['target'] = data.target
        return df
    elif name == "Titanic":
        return sns.load_dataset("titanic").dropna()
    elif name == "Titanic (Full)":
        return sns.load_dataset("titanic")
    elif name == "Tips":
        return sns.load_dataset("tips").dropna()
    elif name == "Penguins":
        return sns.load_dataset("penguins").dropna()
    elif name == "Flights":
        return sns.load_dataset("flights").dropna()
    elif name == "Diamonds":
        return sns.load_dataset("diamonds").dropna()
    elif name == "Planets":
        return sns.load_dataset("planets").dropna()
    elif name == "Car Crashes":
        return sns.load_dataset("car_crashes").dropna()
    elif name == "Exercise":
        return sns.load_dataset("exercise").dropna()
    elif name == "FMRI":
        return sns.load_dataset("fmri").dropna()
    elif name == "Geyser":
        return sns.load_dataset("geyser").dropna()
    elif name == "Anscombe":
        return sns.load_dataset("anscombe")
    elif name == "Brain Networks":
        return sns.load_dataset("brain_networks")
    else:
        return pd.DataFrame()


def load_uploaded_file(uploaded_file):
    """
    Reads uploaded dataset file and returns a pandas DataFrame.
    Supports: CSV, Excel, JSON, Parquet
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            return None  # Unsupported format

        if df.empty or df.shape[1] == 0:
            return None  # No data/columns

        return df

    except Exception as e:
        print(f"Error reading uploaded file: {e}")
        return None
