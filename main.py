from src.utils import data_loader, model

# Load data
data, merged_df = data_loader.load_data_with_metadata("data/raw/Clothing_Shoes_and_Jewelry_5.json", "data/raw/meta_Clothing_Shoes_and_Jewelry.json")

# Train model
algorithm, testset = model.train_model(data)

# Evaluate model
rmse, precision, recall, predictions = model.evaluate_model(algorithm, testset)
print(f"RMSE: {rmse}")

# Get top 10 recommendations for a user
top_n_recommendations = model.get_top_n_recommendations(predictions, n=10)
user_id = 'A2A2WZYLU528RO'
print(top_n_recommendations.get(user_id, []))