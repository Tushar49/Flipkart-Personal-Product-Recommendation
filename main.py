from src.utils import data_loader, model

# Load data
data = data_loader.load_data("data/raw/Clothing_Shoes_and_Jewelry_5.json")

# Train model
algorithm, testset = model.train_model(data)

# Evaluate model
rmse, precision, recall, predictions = model.evaluate_model(algorithm, testset)
print(f"RMSE: {rmse}")

# Get top 10 recommendations for a user
top_n_recommendations = model.get_top_n_recommendations(predictions, n=10)
user_id = 'A2A2WZYLU528RO'
print(top_n_recommendations.get(user_id, []))