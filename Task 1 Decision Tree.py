import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/Company_Data.csv")

data.isnull().sum()
data.dropna()
data.columns

#perform discretization/bucketing using pd.cut 
# converting continuos column into discrete category - Because Decision Tree works on Discrete data
data['Sales']=pd.cut(data['Sales'],2,labels=['high','low'])

data.info()
#converting into binary , converting non numeric data into categorical(binary) data using label encoder 
lb = LabelEncoder()
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

#data["default"]=lb.fit_transform(data["default"])

data['Sales'].unique()
data['Sales'].value_counts()
colnames = list(data.columns)

predictors = colnames[1:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT # for numeric output DecisionTreeRegressor

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(preds, test[target], rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == test[target]) # Test Data Accuracy  - 75%

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(preds, train[target], rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == train[target]) # Train Data Accuracy # 100% accuracy - overfitting case

# overfitting case because train accuracy is high as compared to test accuracy

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)
