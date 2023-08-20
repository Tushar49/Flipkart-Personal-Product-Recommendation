from surprise import Dataset, Reader
import pandas as pd
import json
import ast

def load_data(filepath):
    # Load the dataset into a pandas dataframe
    df = pd.read_json(filepath, lines=True)
    
    # Handle missing values: Drop rows with missing values in the columns of interest
    df = df.dropna(subset=['reviewerID', 'asin', 'overall'])
    
    # Handle outliers: We'll cap the ratings at 1-5 in case there are any errors
    df['overall'] = df['overall'].clip(1, 5)
    
    # Keep only necessary columns and rename them
    data_df = df[['reviewerID', 'asin', 'overall']].rename(columns={
        'reviewerID': 'ReviewerID',
        'asin': 'ASIN',
        'overall': 'Score'
    })

    # Define a Reader. The rating scale is from 1 to 5 in the dataset.
    reader = Reader(rating_scale=(1, 5))

    # Load the data from the dataframe into Surprise's format
    data = Dataset.load_from_df(data_df, reader)
    return data


import json

def load_data_with_metadata(review_filepath, metadata_filepath):
    # Load the review dataset
    review_df = pd.read_json(review_filepath, lines=True, encoding='utf-8')
    
    # Load the product metadata using the robust method
    with open(metadata_filepath, 'r', encoding='utf-8') as file:
        metadata_list = [ast.literal_eval(line) for line in file]
    metadata_df = pd.DataFrame(metadata_list)
    
    # Merge the two datasets on the 'asin' column
    merged_df = pd.merge(review_df, metadata_df, on="asin", how="inner")
    
    # Handle missing values and outliers for the review data
    merged_df = merged_df.dropna(subset=['reviewerID', 'asin', 'overall'])
    merged_df['overall'] = merged_df['overall'].clip(1, 5)
    
    # Keep only necessary columns and rename them for the surprise library
    data_df = merged_df[['reviewerID', 'asin', 'overall']].rename(columns={
        'reviewerID': 'ReviewerID',
        'asin': 'ASIN',
        'overall': 'Score'
    })

    # Define a Reader. The rating scale is from 1 to 5.
    reader = Reader(rating_scale=(1, 5))

    # Load the data from the dataframe into Surprise's format
    data = Dataset.load_from_df(data_df, reader)
    
    return data, merged_df  # Return both the data in Surprise format and the merged dataframe
