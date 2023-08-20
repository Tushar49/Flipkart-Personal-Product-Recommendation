from surprise import Dataset, Reader
import pandas as pd
import json
import ast
import itertools

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

def load_data_with_metadata(review_filepath, metadata_filepath):
    # Load the review dataset
    review_df = pd.read_json(review_filepath, lines=True, encoding='utf-8')
    review_df.set_index('asin', inplace=True)  # Index on 'asin' for efficient merge
    
    # Initialize an empty list to collect chunks of metadata
    metadata_list = []
    
    # Read and process the metadata file in chunks
    chunk_size = 50000  # Adjust based on available memory
    with open(metadata_filepath, 'r', encoding='utf-8') as file:
        while chunk := list(itertools.islice(file, chunk_size)):
            chunk_data = [ast.literal_eval(line) for line in chunk]
            metadata_chunk_df = pd.DataFrame(chunk_data)
            metadata_chunk_df.set_index('asin', inplace=True)  # Index on 'asin'
            metadata_list.append(metadata_chunk_df)

    # Concatenate all chunks into a single DataFrame
    metadata_df = pd.concat(metadata_list)

    # Merge the two datasets on the 'asin' index
    merged_df = pd.merge(review_df, metadata_df, left_index=True, right_index=True, how="inner")
    
    # Generate the data variable using the merged_df
    data_df = merged_df[['reviewerID', 'asin', 'overall']].rename(columns={
        'reviewerID': 'ReviewerID',
        'asin': 'ASIN',
        'overall': 'Score'
    })
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_df, reader)
    
    return data, merged_df  # Return both the data in Surprise format and the merged dataframe
