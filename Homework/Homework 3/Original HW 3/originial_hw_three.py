import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import KNNImputer
from sklearn.utils import shuffle
from sklearn import metrics
from scipy.stats import mode
import matplotlib.pyplot as plt

def preprocess_football_data(data):
    
    data.Date = [int(x.replace("/", "")) for x in data.Date]
    data.Media = [int(x[0]) for x in data.Media]
    data.Label = [0 if x == "Lose" else 1 for x in data.Label]
    data.Is_Home_or_Away = [0 if x == "Away" else 1 for x in data.Is_Home_or_Away]
    data.Is_Opponent_in_AP25_Preseason = [0 if x == "Out" else 1 for x in data.Is_Opponent_in_AP25_Preseason]
    data.Opponent = [(ord(x[0])+ord(x[1])+ord(x[2])) for x in data.Opponent]
    data = data.drop(columns="ID")
    return data

def preprocess_data(train_df):
    # From HW 1
    # Code from Toward Data Science Article
    # https://towardsdatascience.com/missing-value-imputation-with-python-and-k-nearest-neighbors-308e7abd273d
    gender = [0 if sex == "male" else 1 for sex in train_df["Sex"]]
    train_df["Gender"] = gender


    temp_train = train_df.copy()
    sex = train_df.Sex
    temp_train = temp_train.drop(columns="Sex")
    temp_train.Name = [hash(x) for x in temp_train.Name]
    temp_train.Ticket = [hash(x) for x in temp_train.Ticket]
    temp_train.Cabin = [hash(x) for x in temp_train.Cabin]
    emb = []
    for x in temp_train.Embarked:
        if x == "S":
            emb.append(0)
        elif x == "C":
            emb.append(1)
        elif x == "Q":
            emb.append(2)
        else:
            #print("NAN")
            emb.append(0)
            
    temp_train.Embarked = emb
    imputer = KNNImputer(n_neighbors=20)
    imputed = imputer.fit_transform(temp_train)
    train_df = pd.DataFrame(imputed, columns=temp_train.columns)
    train_df["Sex"] = sex

    fares = []
    #print("Mode: ", train_df["Fare"].mode())
    for x in train_df.Fare:
        if x > -0.001 and x <= 7.91:
            fares.append(0)
        elif x > 7.91 and x <= 14.454:
            fares.append(1)
        elif x > 14.454 and x <= 31.0:
            fares.append(2)
        elif x < 31.0 and x <= 512.329:
            fares.append(3)
        else:
            fares.append(1)
    train_df.Fare = fares
    return train_df

#https://scikit-learn.org/stable/modules/naive_bayes.html
def naive_bayes(train, test, xLabels, yLabel):
    assert (type(xLabels) == list), "xLabels must be a list of features"
    assert (type(yLabel) == str), "yLabel must be a string"
    
    xTrain = train[xLabels]
    yTrain = train[yLabel]
    
    xTest = test[xLabels]
    yTest = test[yLabel]
    
    gnb = GaussianNB()
    clf = gnb.fit(xTrain, yTrain)
    
    yPred = clf.predict(xTest)
    acc = metrics.accuracy_score(yTest, yPred)
    prc = metrics.precision_score(yTest, yPred)
    rec = metrics.recall_score(yTest, yPred)
    f1 = metrics.f1_score(yTest, yPred)
    return acc, prc, rec, f1, (yTest, yPred)
    
def knn(train, new_item, yLabel, k=6):
    
    #xTrain = train[xLabels]
    #yTrain = train[yLabel]
    
    #xTest = test[xLabels]
    #yTest = test[yLabel]
    
    train_yes = train.loc[train[yLabel] == 1]
    train_yes = train_yes.drop(columns=yLabel)
    train_no = train.loc[train[yLabel] == 0]
    train_no = train_no.drop(columns=yLabel)
    data = [train_yes, train_no]
    
    distances = []
    for i in range(len(data)):
        for rid, row in data[i].iterrows():
            rData = row.to_numpy()
            dist = np.sqrt(np.sum( (rData - np.array(new_item))**2 ))
            distances.append( [dist, i] )
    
    votes = [i[1] for i in sorted(distances)[:k]]
    prediction = mode(votes)
    #print(votes)
    #print(prediction[0])
    return prediction[0]
    
    
    
    
    
    
if __name__ == '__main__':
    football_train = preprocess_football_data(pd.read_csv("./Football/train.csv"))
    football_test = preprocess_football_data(pd.read_csv("./Football/test.csv"))
    
    #print(football_train)
    #print(football_test)
    
    accuracy, precision, recall, f1, labelTuple  = naive_bayes(football_train, football_test, ["Date", "Opponent", "Is_Home_or_Away", "Is_Opponent_in_AP25_Preseason", "Media"], "Label")
    
    print("Football Naive Bayes:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    
    print("Outcomes\nTrue | Pred")
    for i in range(len(labelTuple[0])):
        gT, yH = labelTuple[0][i], labelTuple[1][i]
        print("  " + str(int(gT)) + "  |  " + str(int(yH)))
        
    print("=============================")
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    labelTuple = ([], [])
    for rid, row in football_test.iterrows():
        rData = row.to_numpy()
        xTest = rData[:-1]
        yTest = rData[-1]
        yHat = knn(football_train, xTest, "Label")[0]
        labelTuple[0].append(yTest)
        labelTuple[1].append(yHat)
        if yHat == yTest and yTest == 1: # TP
            tp += 1
        elif yHat == yTest and yTest == 0: # TN
            tn += 1
        elif yHat != yTest and yTest == 1: # FN
            fn += 1
        else: # FP
            fp += 1
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2*  recall * precision) / (recall + precision) 
    
    print("Football KNN:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    print("Outcomes\nTrue | Pred")
    for i in range(len(labelTuple[0])):
        gT, yH = labelTuple[0][i], labelTuple[1][i]
        print("  " + str(int(gT)) + "  |  " + str(int(yH)))
        
    print("=============================")
            
            
    """
    --------------------------------------------------------
    ========================================================
    --------------------------------------------------------
    """
    
    titanic_train = preprocess_data(pd.read_csv("./Titanic/train.csv"))
    titanic_test = preprocess_data(pd.read_csv("./Titanic/test.csv"))
    #["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked", "Survived"]])
    
    avgs = [0, 0, 0, 0]
    for i in range(5):
        train = titanic_train.copy()
        train = shuffle(train)
        train, test = train[:int(len(titanic_train)*.8)], train[int(len(titanic_train)*.8):]
        
        accuracy, precision, recall, f1, labelTuple  = naive_bayes(titanic_train[:int(len(titanic_train)*.75)], titanic_train[int(len(titanic_train)*.75):], ["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked"], "Survived")
        avgs[0] += accuracy
        avgs[1] += precision
        avgs[2] += recall 
        avgs[3] += f1
    
    
    print("Titanic Naive Bayes Five-Fold:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}".format(avgs[0]/5, avgs[1]/5, avgs[2]/5, avgs[3]/5))
   
    print("=============================")
    
    Y = [] # KNN Accuracies
    X = [] # Value of K
    Ttrain = titanic_train[:int(len(titanic_train)*.75)]
    Ttest = titanic_train[int(len(titanic_train)*.75):]
    for k_val in range(3, 100, 2):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for rid, row in Ttest[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].iterrows():
            rData = row.to_numpy()
            xTest = rData[:-1]
            yTest = rData[-1]

            yHat = knn(Ttrain[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked", "Survived"]], xTest, "Survived", k=k_val)[0]
            if yHat == yTest and yTest == 1: # TP
                tp += 1
            elif yHat == yTest and yTest == 0: # TN
                tn += 1
            elif yHat != yTest and yTest == 1: # FN
                fn += 1
            else: # FP
                fp += 1
        #print(tp, tn, fp, fn)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        Y.append(accuracy)
        X.append(k_val)
    
    print("Titanic KNN:\n")
    plt.plot(X, Y)
    plt.title("KNN Accuracy V. K")
    plt.xlabel("K (Integer Value)")
    plt.ylabel("KNN Accuracy")
    plt.show()
    

    print("=============================")
