import ast
import dask.dataframe as dd
import json
import pandas as pd

def convert_to_valid_json(json_string):
    """Convert a JSON string with single quotes to one with double quotes."""
    return json_string.replace('\'', '\"')

def load_metadata_in_chunks_improved(filepath, chunksize=50000):
    chunks = []
    error_lines = []
    line_number = 0
    
    with open(filepath, 'r') as file:
        lines = []
        for line in file:
            line_number += 1
            try:
                # Attempt to parse the line using ast.literal_eval
                parsed_line = ast.literal_eval(line)
                valid_json_line = json.dumps(parsed_line)
                lines.append(valid_json_line)
                if len(lines) == chunksize:
                    valid_chunk = dd.from_pandas(pd.read_json('\n'.join(lines), lines=True), npartitions=1)
                    chunks.append(valid_chunk)
                    lines.clear()
            except Exception:
                error_lines.append(line_number)

        # Handle the last chunk if it exists
        if lines:
            valid_chunk = dd.from_pandas(pd.read_json('\n'.join(lines), lines=True), npartitions=1)
            chunks.append(valid_chunk)
    
    if error_lines:
        print(f"Skipped lines due to errors: {error_lines}")
        
    return dd.concat(chunks, interleave_partitions=True)

def load_data_with_metadata(review_filepath, metadata_filepath):
    # Load the review data
    review_df = dd.read_json(review_filepath, lines=True, encoding='utf-8')
    # Load the metadata
    metadata_df = load_metadata_in_chunks_improved(metadata_filepath)
    # Merge the datasets
    merged_df = dd.merge(review_df, metadata_df, on="asin", how="inner")
    # Return the datasets
    return review_df, merged_df
