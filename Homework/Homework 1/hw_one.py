import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


train_df = pd.read_csv("./Titanic/train.csv")
test_df = pd.read_csv("./Titanic/test.csv")

combined_data = [train_df, test_df]


for col in train_df:
    print(train_df[col].describe())
    print("=======================")
    print(train_df[col].value_counts())
    print("++++++++++++++++++++++")
    
print("\n\n-----------------------")
class_cross_survive = train_df.groupby("Pclass")["Survived"].mean()
print("Class Correlation: \n", class_cross_survive)
print("****")

sex_cross_survive = train_df.groupby("Sex")["Survived"].mean()
print("Sex Correlation: \n", sex_cross_survive)
print("****")


print(train_df[(train_df["Survived"] == 1)]["Age"])
did_surv = train_df[(train_df["Survived"] == 1)]["Age"]
not_surv = train_df[(train_df["Survived"] == 0)]["Age"]

fig, axes = plt.subplots(1, 2)
did_surv.hist(ax=axes[0])
axes[0].set_title("Survived = 1")
axes[0].set_xlabel("Age")
not_surv.hist(ax=axes[1])
axes[1].set_title("Survived = 0")
axes[1].set_xlabel("Age")
plt.show()


did_pc1 = train_df.loc[(train_df["Survived"] == 1) & (train_df["Pclass"]==1), ["Age"]]
did_pc2 = train_df.loc[(train_df["Survived"] == 1) & (train_df["Pclass"]==2), ["Age"]]
did_pc3 = train_df.loc[(train_df["Survived"] == 1) & (train_df["Pclass"]==3), ["Age"]]

not_pc1 = train_df.loc[(train_df["Survived"] == 0) & (train_df["Pclass"]==1), ["Age"]]
not_pc2 = train_df.loc[(train_df["Survived"] == 0) & (train_df["Pclass"]==2), ["Age"]]
not_pc3 = train_df.loc[(train_df["Survived"] == 0) & (train_df["Pclass"]==3), ["Age"]]

fig, axes = plt.subplots(3, 2, sharex=True)

did_pc1.hist(ax=axes[0,0])
axes[0,0].set_title("Pclass = 1 | Survived = 1")

did_pc2.hist(ax=axes[1,0])
axes[1,0].set_title("Pclass = 2 | Survived = 1")

did_pc2.hist(ax=axes[2,0])
axes[2,0].set_title("Pclass = 3 | Survived = 1")
axes[2,0].set_xlabel("Age")

not_pc1.hist(ax=axes[0,1])
axes[0,1].set_title("Pclass = 1 | Survived = 0")

not_pc2.hist(ax=axes[1,1])
axes[1,1].set_title("Pclass = 2 | Survived = 0")

not_pc3.hist(ax=axes[2,1])
axes[2,1].set_title("Pclass = 3 | Survived = 0")
axes[2,1].set_xlabel("Age")

plt.show()

did_s = train_df.loc[(train_df["Survived"] == 1) & (train_df["Embarked"] == "S"), ["Sex", "Fare"]]
did_s = did_s.groupby("Sex")["Fare"].mean()
did_c = train_df.loc[(train_df["Survived"] == 1) & (train_df["Embarked"] == "C"), ["Sex", "Fare"]]
did_c = did_c.groupby("Sex")["Fare"].mean()
did_q = train_df.loc[(train_df["Survived"] == 1) & (train_df["Embarked"] == "Q"), ["Sex", "Fare"]]
did_q = did_q.groupby("Sex")["Fare"].mean()

not_s = train_df.loc[(train_df["Survived"] == 0) & (train_df["Embarked"] == "S"), ["Sex", "Fare"]]
not_s= not_s.groupby("Sex")["Fare"].mean()
not_c = train_df.loc[(train_df["Survived"] == 0) & (train_df["Embarked"] == "C"), ["Sex", "Fare"]]
not_c = not_c.groupby("Sex")["Fare"].mean()
not_q = train_df.loc[(train_df["Survived"] == 0) & (train_df["Embarked"] == "Q"), ["Sex", "Fare"]]
not_q = not_q.groupby("Sex")["Fare"].mean()

print(not_q)


fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)

did_s.plot.bar(ax=axes[0,0])
axes[0,0].set_title("Embarked = S | Survived = 1")
axes[0,0].set_ylabel("Fare")

did_c.plot.bar(ax=axes[1,0])
axes[1,0].set_title("Embarked = C | Survived = 1")
axes[1,0].set_ylabel("Fare")

did_q.plot.bar(ax=axes[2,0])
axes[2,0].set_title("Embarked = Q | Survived = 1")
axes[2,0].set_xlabel("Sex")
axes[2,0].set_ylabel("Fare")

not_s.plot.bar(ax=axes[0,1])
axes[0,1].set_title("Embarked = S | Survived = 0")

not_c.plot.bar(ax=axes[1,1])
axes[1,1].set_title("Embarked = C | Survived = 0")

not_q.plot.bar(ax=axes[2,1])
axes[2,1].set_title("Embarked = Q | Survived = 0")
axes[2,1].set_xlabel("Sex")

plt.show()


did_ticket = train_df[(train_df["Survived"] == 1)]["Ticket"]
not_ticket = train_df[(train_df["Survived"] == 0)]["Ticket"]

fig, axes = plt.subplots(1, 2)
did_ticket.hist(ax=axes[0])
axes[0].set_title("Survived = 1")
axes[0].set_xlabel("Ticket")
axes[0].set_xticklabels(axes[0].get_xticks(), rotation=45)
not_ticket.hist(ax=axes[1])
axes[1].set_title("Survived = 0")
axes[1].set_xlabel("Ticket")
axes[1].set_xticklabels(axes[1].get_xticks(), rotation=45)
plt.show()    


print(train_df["Cabin"].count())
print(test_df["Cabin"].count())
print(len(combined_data[0]), len(combined_data[1]))


gender = [0 if sex == "male" else 1 for sex in train_df["Sex"]]
train_df["Gender"] = gender
gender = [0 if sex == "male" else 1 for sex in test_df["Sex"]]
test_df["Gender"] = gender

# Code from Toward Data Science Article
# https://towardsdatascience.com/missing-value-imputation-with-python-and-k-nearest-neighbors-308e7abd273d
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
        print("NAN")
        emb.append(0)
        
temp_train.Embarked = emb
imputer = KNNImputer(n_neighbors=20)
imputed = imputer.fit_transform(temp_train)
train_df = pd.DataFrame(imputed, columns=temp_train.columns)
train_df["Sex"] = sex

emb = []
for x in temp_train.Embarked:
    if x == 0:
        emb.append("S")
    elif x == 1:
        emb.append("C")
    else:
        emb.append("Q")
train_df.Embarked = emb

fares = []
print("Mode: ", train_df["Fare"].mode())
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
print(train_df)


