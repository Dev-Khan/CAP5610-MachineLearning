import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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
    
    """
    emb = []
    for x in temp_train.Embarked:
        if x == 0:
            emb.append("S")
        elif x == 1:
            emb.append("C")
        else:
            emb.append("Q")
    train_df.Embarked = emb
    """

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

def create_five_fold_crossentropy(train, test):
    train.append(test)
    rows, col = train.shape
    split = np.linspace(0, rows, num=6)
    data = []
    for i in range(len(split)):
        #print(int(split[i]))
        if i != 0:
            data.append(train.iloc[int(split[i-1]):int(split[i])])
    return data

def select_features(data):
    # Base on HW 1 we are dropping 
    # age, ticket, and cabine
    # Name is being dropped to help reduce complexity of the model
    # as all names are unique and don't offere any intrinsic information
    # As well PassengerID, since it is only an indexing method and has no
    # correlation to survival rate
    # Drop sex since it has been replaced with Gender
    return data.drop(columns=["Age", "Ticket", "Cabin", "Name", "PassengerId", "Sex"])

def make_decision_tree(data, test, target="Survived"):
    
    tree = dict()
    
    def gini_split(data, col, label):
        total = data[col].count()
        counts = data[col].value_counts()
        #print(counts)
        #print("------")
        percent = data.groupby(col)[target].value_counts(normalize=True)
        #print(percent[0])
        #print(percent[1])
        #print(percent[2])
        #print(len(percent))
        #print("-----")
        gini = []
        for i in range(len(percent)):
            if not isinstance(percent[i], np.float64):
                g = 1
                #print(percent[i])
                for p in percent[i]:
                    g -= p**2
                g *= (counts[i]/total)
                gini.append(g)
        return sum(gini)
    
    def recusrive_call(data, key):
        # Steps:
        # ID best split
        # Take each path of split (recusive)
        # If at target as only label, leaf node; return
        
        # If only target left, base case
        if data.shape[1] ==1 and (target in data.columns):
            outcome = data[target].value_counts(normalize=True)
            tree[key] = [outcome[0], outcome[1]]
            #print("Hit BASE CASE")
            return
            
        # ID Best split
        maxGini = 0
        maxCol = None
        for col in data:
            if col != target:
                gini = gini_split(data, col, target)
                if maxGini < gini:
                    maxGini = gini
                    maxCol = col
                #print("Col: {} | Gini: {}".format(col, gini))
        #print("MAX GINI: {}\nCOL: {}".format(maxGini, maxCol))
        
        # Build tree
        dc = data.copy()
        dc = dc.drop(columns=maxCol)
        if key == 0:
            key = maxCol
            tree["root"] = maxCol
        tree[key] = []
        for val in data[maxCol].unique():
            #print(val)
            key_name = "{}_{}_a".format(maxCol, val)
            while key_name in tree:
                key_name =key_name[:-1] + chr(ord(key_name[-1]) + 1)
                #print(key_name)
            #print(key_name)
            tree[key].append(key_name)
            recusrive_call(dc, key_name)
        
    
    recusrive_call(data, 0)
    return tree

def traverse_tree(tree, data):
    # get root start
    value = data[tree["root"]]
    #print(value)
    key = "{}_{}_a".format(tree["root"], value)
    #print(key)
    
    # Do this "recusively"
    while not isinstance(tree[key][0], np.float64):
        paths = tree[key]
        col, _, uuid = paths[0].split("_")
        value = data[col]
        if (col == "Fare"):
            value = int(value)
        key = "{}_{}_{}".format(col, value, uuid)
        #print(key)
    outcome = tree[key]
    if outcome[0] > outcome[1]:
        return 0 # Did not survive
    else:
        return 1 # Survived
    
def print_tree(tree):
    size = 4
    print(".root")
    def dfs(tree, key, depth, pipes):
        if isinstance(tree[key][0], np.float64): 
            string = "|   "*depth + "+" + "-"*size
            string += key.split("_")[0]
            print(string)
            return
        
        string = "|   "*depth + pipes.pop(0) + "-"*size
        string += key.split("_")[0]
        print(string)
        _pipes = ["|" for x in range(len(tree[key]))]
        _pipes[-1] = "+"
        for key_v in tree[key]:
            dfs(tree, key_v, depth+1, _pipes)
        
    pipes = ["|" for x in range(len(tree[tree['root']]))]
    pipes[-1] = "+"
    for key_main in tree[tree['root']]:
        #print("{}--{}".format(pipes.pop(0), key_main))
        dfs(tree, key_main, 0, pipes) 
        
def doCrossValidationDT(data, foldCount=5):
    avg_acc = 0
    for fold in range(foldCount):
        # split data into [4 train][1 test]
        temp = data[fold+1:] + data[:fold]
        train = pd.DataFrame(temp[0])
        for i in range(1, len(temp)):
            train.append(temp[i])
        print(train)
        test = data[fold]
        # make tree
        T = make_decision_tree(train, None)
        # test tree
        total = 0
        correct = 0
        for rid, row in test.iterrows():
            try:
                if traverse_tree(T, row) == row["Survived"]:
                    correct += 1
            except KeyError as ke:
                print("Could not classify value. FAILED")
            finally:
                total += 1
        # return accuracy for fold
        avg_acc += (correct/total)
    avg_acc /= foldCount
    return avg_acc

def doCrossValidationRFC(data, foldCount=5):
    avg_acc = 0
    for fold in range(foldCount):
        # split data into [4 train][1 test]
        temp = data[fold+1:] + data[:fold]
        train = pd.DataFrame(temp[0])
        for i in range(1, len(temp)):
            train.append(temp[i])
        print(train)
        xTrain = train[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked"]]
        yTrain = train["Survived"]
        test = data[fold]
        xTest = test[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked"]]
        yTest = test["Survived"] 
        # make random forest
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(xTrain, yTrain)
        # test random forest
        yPred = clf.predict(xTest)
        avg_acc += metrics.accuracy_score(yTest, yPred)
        # return accuracy for fold
    avg_acc /= foldCount
    return avg_acc
    
if __name__ == '__main__':
    train_df = preprocess_data(pd.read_csv("./Titanic/train.csv"))
    test_df = preprocess_data(pd.read_csv("./Titanic/test.csv"))
    
    train_df = select_features(train_df)
    test_df = select_features(test_df)
    
    fiveFoldData = create_five_fold_crossentropy(train_df, test_df)
    #print(fiveFoldData[0])
    
    avg_acc = doCrossValidationDT(fiveFoldData)
    print("Decision Tree Avg Accuracy: ", avg_acc) # ~47.2%
    avg_acc = doCrossValidationRFC(fiveFoldData)
    print("Random Forest Avg Accuracy:" ,avg_acc) # ~78.6%
    
    
    #T = make_decision_tree(train_df, None)
    #print_tree(T)
    
    #traverse_tree(T, test_df.iloc[0])
