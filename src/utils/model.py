import os
import pickle

from surprise import SVD, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, NormalPredictor, CoClustering
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import accuracy

def train_model(data):
    # Split the data into training and test sets

    # Check if a saved model exists
    if os.path.exists('saved_model.pkl'):
        with open('saved_model.pkl', 'rb') as file:
            algorithm = pickle.load(file)
        return algorithm, testset
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Define a list of algorithms to try
    algorithms = [SVD(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), NMF(), NormalPredictor(), CoClustering()]
    
    # Hyperparameter tuning (for the sake of simplicity, we'll only do this for SVD here)
    
param_grid = {
    'n_epochs': [10, 20],
    'lr_all': [0.005],
    'reg_all': [0.2, 0.4]
}

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    

    # Save the trained model

    # Best algorithm and parameters
    best_algorithm = gs.best_estimator['rmse']
    
    # Train the model
    best_algorithm.fit(trainset)
    with open('saved_model.pkl', 'wb') as file:
        pickle.dump(best_algorithm, file)
    return best_algorithm, testset

# def evaluate_model(algorithm, testset):
#     predictions = algorithm.test(testset)
    
#     # Compute and print RMSE (Root Mean Squared Error)
#     rmse = accuracy.rmse(predictions)
    
#     # Additional metrics (precision and recall)
#     precisions, recalls = accuracy.precision_recall_at_k(predictions, k=5, threshold=4)
#     precision = sum(prec for prec in precisions.values()) / len(precisions)
#     recall = sum(rec for rec in recalls.values()) / len(recalls)
    
#     return rmse, precision, recall, predictions

def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, []).append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user."""

    # First map the predictions to each user.
    user_est_true = dict()
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

# We'll now integrate this function into the model.py's evaluate_model function.

# For simplicity, we'll return a modified version of the evaluate_model function.
def evaluate_model(algorithm, testset, k=5, threshold=4):
    predictions = algorithm.test(testset)
    
    # Compute and print RMSE (Root Mean Squared Error)
    rmse = accuracy.rmse(predictions)
    
    # Additional metrics (precision and recall)
    precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    
    return rmse, precision, recall, predictions

# Return the modified function for review
# modified_evaluate_model