#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold


titanic = pd.read_csv('data/train.csv')
titanic_test = pd.read_csv('data/test.csv')

print(titanic.info())
print()
print(titanic.isnull().sum())
print()
print(titanic.head(5)["Sex"])
print()

# print(train.columns)
# print(train.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic["Sex"].unique())
print()
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

print(titanic["Embarked"].unique())
print()
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# alg = LinearRegression()
#
# kf = KFold(titanic.shape[0], n_folds=10, random_state=1)
# predictions = []
# for train, test in kf:
#     # 提取出用作训练的数据行（不含拟合目标）
#     train_predictors = (titanic[predictors].iloc[train, :])
#     # 提取用于训练的拟合目标
#     train_target = titanic["Survived"].iloc[train]
#     # 基于训练数据和拟合目标训练模型
#     alg.fit(train_predictors, train_target)
#     # 接下来在测试集上执行预测
#     test_predictions = alg.predict(titanic[predictors].iloc[test, :])
#     predictions.append(test_predictions)
#
#
#
#
# # axis=0 因为现在数组只有一维
# predictions = np.concatenate(predictions, axis=0)
#
# # 将浮点数结果映射为二进制结果（0/1表示幸存与否）
# predictions[predictions > .5] = 1
# predictions[predictions <= .5] = 0
# accuracy = len(predictions[predictions == titanic["Survived"]]) / len(predictions)
# print(accuracy)
# print()
#
#
# alg = LogisticRegression(random_state=1)
# scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# print(scores.mean())
# print()




'''
Submission result
'''
print(titanic_test.isnull().sum())
print()
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

alg = LogisticRegression(random_state=1)
alg.fit(titanic[predictors], titanic["Survived"])
# print(type(titanic_test[predictors]))
predictions = alg.predict(titanic_test[predictors])

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("data/result.csv", index=False)