from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_model(data, algorithm=SVD()):
    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # Train the model
    algorithm.fit(trainset)
    return algorithm, testset

def evaluate_model(algorithm, testset):
    predictions = algorithm.test(testset)
    # Compute and print RMSE (Root Mean Squared Error)
    rmse = accuracy.rmse(predictions)
    return rmse, predictions

def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, []).append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
