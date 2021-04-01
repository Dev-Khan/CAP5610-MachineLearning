from surprise import SVD, KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

def load_data():
    df = pd.read_csv("./archive/ratings_small.csv")
    df.columns=['userID', 'movieID', 'rating', 'timestamp']
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['userID', 'movieID', 'rating']], reader)
    print(data)
    return data

def build_recommender_system(algo, data):
    return cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def task_a_to_e():
    rating_data = load_data()
    
    # Questions c, d, e are below
    performace = build_recommender_system(SVD(), rating_data)
    
    
    #MSD is default
    msd_mve = {'name':'msd', 'user-based':False}
    
    cosine_usr = {'name':'cosine', 'user-based':True}
    cosine_mve = {'name':'cosine', 'user-based':False}
    
    pearson_usr = {'name':'pearson_baseline', 'user-based':True}
    pearson_mve = {'name':'pearson_baseline', 'user-based':False}
    
    
    usr = {"cosine": 0, "msd": 0, "pearson": 0}
    mve = {"cosine": 0, "msd": 0, "pearson": 0}
    
    print("MSD user")
    _ = build_recommender_system(KNNBasic(), rating_data)
    #usr["msd"] = sum(_["test_rmse"])/5
    print(_)
    usr["msd"] = sum(_["test_mae"])/5
    
    print("MSD Movie")
    _ = build_recommender_system(KNNBasic(sim_options=msd_mve), rating_data)
    #mve["msd"] = sum(_["test_rmse"])/5
    mve["msd"] = sum(_["test_mae"])/5
    
    print("Cosine User")
    _ = build_recommender_system(KNNBasic(sim_options=cosine_usr), rating_data)
    #usr["cosine"] = sum(_["test_rmse"])/5
    usr["cosine"] = sum(_["test_mae"])/5
    
    print("Cosine Movie")
    _ = build_recommender_system(KNNBasic(sim_options=cosine_mve), rating_data)
    #mve["cosine"] = sum(_["test_rmse"])/5
    mve["cosine"] = sum(_["test_mae"])/5
    
    print("Pearson User")
    _ = build_recommender_system(KNNBasic(sim_options=pearson_usr), rating_data)
    #usr["pearson"] = sum(_["test_rmse"])/5
    usr["pearson"] = sum(_["test_mae"])/5
    
    print("Pearson Movie")
    _ = build_recommender_system(KNNBasic(sim_options=pearson_mve), rating_data)
    #mve["pearson"] = sum(_["test_rmse"])/5
    mve["pearson"] = sum(_["test_mae"])/5
    
    fig, ax = plt.subplots()
    y = np.arange(len(usr))
    width=0.4
    ax.bar(y + width/2, list(usr.values()), label="User CF", color="#0000FF")
    ax.bar(y - width/2, list(mve.values()), label="Movie CF",  color="#FF0000")
    #ax.set_ylabel("RMSE")
    ax.set_ylabel('MAE')
    ax.set_xticklabels(["MSD", "Cosine", "Pearson"])
    ax.legend()
    plt.xticks(np.arange(min(y), max(y)+1, 1.0))
    plt.show()
    
def task_f(usrBased=True):
    rating_data = load_data()
    # Question f (number of neighbors)
    
    X = []
    RMSE = []
    MAE = []
    for k in range(10, 100, 10):
        msd = {'name':'msd', 'user-based':usrBased}
        _ = build_recommender_system(KNNBasic(sim_options=msd, k=k), rating_data)
        X.append(k)
        RMSE.append(sum(_["test_rmse"])/5)
        MAE.append(sum(_["test_mae"])/5)

    fig, ax = plt.subplots()
    ax.plot(X, RMSE, label="RMSE")
    ax.plot(X, MAE, label="MAE")
    ax.set_xlabel("K")
    ax.set_ylabel("Value")
    ax.legend()
    t = "User CF" if usrBased else "Movie CF"
    plt.title(t)
    plt.show()
    
if __name__ == '__main__':
    
    task_f(usrBased=False)
    
        
    
    # Quation g (best k value)
    
