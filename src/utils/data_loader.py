from surprise import Dataset, Reader
import pandas as pd
import json
import ast
import itertools
import gc


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
    # Define a Reader. The rating scale is from 1 to 5 in the dataset.
    reader = Reader(rating_scale=(1, 5))

    # Adjust chunk size based on available memory
    review_chunk_size = 1000000
    metadata_chunk_size = 2000000

    # Lists to collect chunks
    all_data_chunks = []
    all_metadata_chunks = []

    # Process review data in chunks
    review_chunk_iter = pd.read_json(review_filepath, lines=True, chunksize=review_chunk_size, encoding='utf-8')
    for review_chunk in review_chunk_iter:
        review_chunk.set_index('asin', inplace=True)  # Index on 'asin' for efficient merge

        # Read and process the metadata file in chunks
        with open(metadata_filepath, 'r', encoding='utf-8') as file:
            for chunk in itertools.islice(file, metadata_chunk_size):
                chunk_data = ast.literal_eval(chunk)
                metadata_chunk = pd.DataFrame([chunk_data])
                metadata_chunk.set_index('asin', inplace=True)  # Index on 'asin'
                
                # Merge with review chunk
                merged_chunk = pd.merge(review_chunk, metadata_chunk, left_index=True, right_index=True, how="inner")
                merged_chunk_reset = merged_chunk.reset_index()

                # Append to lists
                all_data_chunks.append(merged_chunk_reset[['reviewerID', 'asin', 'overall']])
                all_metadata_chunks.append(merged_chunk_reset)
                
                # # Merge with review chunk
                # merged_chunk = pd.merge(review_chunk, metadata_chunk, left_index=True, right_index=True, how="inner")
                
                # # Append to lists
                # all_data_chunks.append(merged_chunk[['reviewerID', 'asin', 'overall']])
                # all_metadata_chunks.append(merged_chunk)

                # Explicitly release memory
                del metadata_chunk
                gc.collect()

        # Explicitly release memory
        del review_chunk
        gc.collect()

    # Concatenate chunks
    data_df = pd.concat(all_data_chunks)
    merged_df = pd.concat(all_metadata_chunks)

    # Load the data into Surprise's format
    data = Dataset.load_from_df(data_df, reader)
    
    return data, merged_df