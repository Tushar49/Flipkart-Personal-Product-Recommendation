
from surprise import SVD, KNNBasic, KNNWithMeans, KNNWithZScore, NMF, NormalPredictor, CoClustering
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import accuracy

def train_model(data):
    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Define a list of algorithms to try
    algorithms = [SVD(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), NMF(), NormalPredictor(), CoClustering()]
    
    # Hyperparameter tuning (for the sake of simplicity, we'll only do this for SVD here)
    param_grid = {
        'n_epochs': [5, 10, 20],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.2, 0.4, 0.6]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    
    # Best algorithm and parameters
    best_algorithm = gs.best_estimator['rmse']
    
    # Train the model
    best_algorithm.fit(trainset)
    return best_algorithm, testset

def evaluate_model(algorithm, testset):
    predictions = algorithm.test(testset)
    
    # Compute and print RMSE (Root Mean Squared Error)
    rmse = accuracy.rmse(predictions)
    
    # Additional metrics (precision and recall)
    precisions, recalls = accuracy.precision_recall_at_k(predictions, k=5, threshold=4)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    
    return rmse, precision, recall, predictions

def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, []).append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
