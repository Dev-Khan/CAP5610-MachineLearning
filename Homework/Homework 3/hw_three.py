import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import KNNImputer
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



def task_two():
    x = [
            [-1,-1*-1],
            [1,1*1],
            [-1, 1*-1],
            [1,-1*1]
        ]
    y = [-1, -1, 1, 1]
    
    pos_x = [-1, 1*-1]
    pos_y = [1, -1*1]
    neg_x = [-1, 1*-1]
    neg_y = [-1, 1*-1]
    
    plt.scatter(pos_x, pos_y, color="blue", label="Positive")
    plt.scatter(neg_x, neg_y, color="red", label="Negative")
    plt.legend()
    plt.ylabel("x1*x2")
    plt.xlabel("x1")
    #plt.show()
    
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.title("Task Two SVM")
    plt.show()
                                                                               
def task_five():
    x = [
            [1,1],
            [2,2],
            [2,0],
            [0,0],
            [1,0],
            [0,1]
        ]
    y = [1, 1, 1, -1, -1, -1]
    
    pos_x = [1, 2, 2]
    pos_y = [1, 2, 0]
    neg_x = [0, 1, 0]
    neg_y = [0, 0, 1]
    
    plt.scatter(pos_x, pos_y, color="blue", label="Positive")
    plt.scatter(neg_x, neg_y, color="red", label="Negative")
    plt.legend()
    plt.ylabel("x2")
    plt.xlabel("x1")
    #plt.show()
    
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.title("Task Five (b) SVM")
    plt.show()
    
def task_six():
    x = [
            [0, np.sqrt(2)*0, 0**2],
            [-1, np.sqrt(2)*-1, -1**2],
            [1, np.sqrt(2)*1, 1**2]
        ]
    y = [1, 1, -1]
    print(x[1][0])
    pos_x = [x[0][0], x[1][0]]
    pos_y = [x[0][1], x[1][1]]
    pos_z = [x[0][2], x[1][2]]
    neg_x = [x[2][0]]
    neg_y = [x[2][1]]
    neg_z = [x[2][2]]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.scatter(pos_x, pos_y, pos_z, color="blue", label="Positive")
    ax.scatter(neg_x, neg_y, neg_z, color="red", label="Negative")
    plt.legend()
    ax.set_ylabel(r"$\sqrt{2} * x_{1}$")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_zlabel(r"$x_{1}^2$")
    #plt.show()
    
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    zz = np.linspace(zlim[0], zlim[1], 30)
    
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = lambda x, y: (-clf.intercept_[0]-clf.coef_[0][0]*x - clf.coef_[0][1]*y) / clf.coef_[0][2]
    #clf.decision_function(xy).reshape(XX.shape)
    print(Z)
    ax.plot_surface(XX, YY, Z(XX, YY), cmap="binary")
    #ax.contour3D(Z[:,0], Z[:,1], Z[:,2], cmap='binary')#colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.title("Task Six (b) SVM")
    plt.show()
    

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

def task_seven():
    titanic_train = preprocess_data(pd.read_csv("./Titanic/train.csv"))
    titanic_test = preprocess_data(pd.read_csv("./Titanic/test.csv"))
    X = titanic_train[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Embarked"]]
    y = titanic_train["Survived"]
    
    kf = KFold(n_splits=5, shuffle=True)
    
    linear_svm = svm.SVC(kernel='linear')
    lin_res = cross_val_score(linear_svm, X, y, cv=kf)
    print("Linar Avg Acc: {}".format(lin_res.mean()))
    
    ploy_svm = svm.SVC(kernel='poly')
    poly_res = cross_val_score(ploy_svm, X, y, cv=kf)
    print("Quadratic Avg Acc: {}".format(poly_res.mean()))
    
    rbf_svm = svm.SVC(kernel="rbf")
    rbf_res = cross_val_score(rbf_svm, X, y, cv=kf)
    print("RBF Avg Acc: {}".format(rbf_res.mean()))
    
if __name__ == '__main__':
    #task_two()
    #task_five()
    #task_six()
    task_seven()
