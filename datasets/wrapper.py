import pandas as pd


def get_weather_data(filepath):
    # import into pandas.
    # If file is too large, then refactor this line to optimize (see helpers.csv_loaders.py)
    df = pd.read_csv(filepath, dtype={'windy':str})

    # one-hot encode categorical variables
    one_hot_data = pd.get_dummies(df[['outlook','windy']])

    # separate the features from the labels
    features_df = df[['temperature','humidity']].join(one_hot_data)
    labels = df['play'].tolist()

    # return the features and labels
    return features_df, labels