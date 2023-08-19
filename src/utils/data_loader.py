from surprise import Dataset, Reader
import pandas as pd

def load_data(filepath):
    # Load the dataset into a pandas dataframe
    df = pd.read_json(filepath, lines=True)

    # Keep only necessary columns
    data_df = df[['ReviewerID', 'ASIN', 'Score']]

    # Define a Reader. The rating scale is from 1 to 5 in the dataset.
    reader = Reader(rating_scale=(1, 5))

    # Load the data from the dataframe into Surprise's format
    data = Dataset.load_from_df(data_df, reader)
    return data