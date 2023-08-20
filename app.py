import streamlit as st
from src.utils import data_loader, model

# Load data with metadata and train model (This should ideally be cached or precomputed for efficiency)
@st.cache(allow_output_mutation=True)
def load_and_train():
    data, merged_df = data_loader.load_data_with_metadata("data/raw/Clothing_Shoes_and_Jewelry_5.json", "data/raw/meta_Clothing_Shoes_and_Jewelry.json")
    algorithm, testset = model.train_model(data)
    return data, merged_df, algorithm, testset

data, merged_df, algorithm, testset = load_and_train()

product_lookup = {row['ASIN']: {'title': row['title'], 'imUrl': row['imUrl'], 'price': row['price']} for index, row in merged_df.iterrows()}


# Streamlit app
st.title("Product Recommendation System")

# User input for user ID
user_id = st.text_input("Enter your User ID:")

# Fetch and display recommendations when user ID is provided
if user_id:
    _, _, _, predictions = model.evaluate_model(algorithm, testset)
    top_n_recommendations = model.get_top_n_recommendations(predictions, n=10)
    recommendations = top_n_recommendations.get(user_id, [])
    
    # Display recommendations with product details
    if recommendations:
        st.header(f"Top Recommendations for User {user_id}")
        for asin, score in recommendations:
            product_data = product_lookup[asin]
            product_name = product_data['title']
            product_image = product_data['imUrl']
            product_price = product_data['price']
            
            st.write(f"Product Name: {product_name}, Price: ${product_price:.2f}")
            st.image(product_image, use_column_width=True)
    else:
        st.write("No recommendations found for the given User ID.")