import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns

#perform discretization/bucketing using pd.cut 
# converting all the continuos columns into discrete categories - Because Decision Tree works on Discrete data
data[' Body mass index']=pd.cut(data[' Body mass index'],3,labels=['high','medium','low'])
data[' Diabetes pedigree function']=pd.cut(data[' Diabetes pedigree function'],3,labels=['high','medium','low'])

data.info()
#converting into binary , converting non numeric data into categorical(binary) data using label encoder 
lb = LabelEncoder()
data[" Body mass index"] = lb.fit_transform(data[" Body mass index"])
data[" Diabetes pedigree function"] = lb.fit_transform(data[" Diabetes pedigree function"])

data.info()

#data["default"]=lb.fit_transform(data["default"])

data[' Class variable'].unique()
data[' Class variable'].value_counts()
colnames = list(data.columns)

#split input and output
predictors = colnames[0:8]
target = colnames[8]

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

np.mean(preds == test[target]) # Test Data Accuracy  - 72%

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(preds, train[target], rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == train[target]) # Train Data Accuracy # 100% accuracy - overfitting case

# overfitting case because train accuracy is high as compared to test accuracy

