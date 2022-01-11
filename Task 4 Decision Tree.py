import pandas as pd
import numpy as np

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns

#perform discretization/bucketing using pd.cut 
# converting Taxable.Income columns into categorical variable
#taxable_income <= 70000 as "genuine" and others are "fake".
data[" monthly income of employee"] = pd.cut(data[" monthly income of employee"], bins = [39343,70000,122391], labels = ["Genuine", "fake"])
data["no of Years of Experience of employee"] = pd.cut(data["no of Years of Experience of employee"], bins = [1,5,10.5], labels = ["valid", "Invalid"])

data.isnull().sum()
data.dropna()
data.info()

#converting into binary , converting non numeric data into categorical(binary) data using get dummies
data = pd.get_dummies(data,columns = ["no of Years of Experience of employee"," monthly income of employee"],drop_first=True)

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
data["Position of the employee"] = lb.fit_transform(data["Position of the employee"])

data.info()

#data["default"]=lb.fit_transform(data["default"])

data[" monthly income of employee_fake"].unique()
data[' monthly income of employee_fake'].value_counts()

#split input and output
predictors = data.drop([' monthly income of employee_fake'], axis=1)
target = data[' monthly income of employee_fake']

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=1) # 70% training and 30% test

from sklearn.tree import DecisionTreeClassifier as DT # for numeric output DecisionTreeRegressor

help(DT)
model = DT(criterion = 'entropy')
model.fit(X_train, y_train)

# Prediction on Test Data
preds = model.predict(X_test)
pd.crosstab(preds, y_test, rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == y_test) # Test Data Accuracy  - 95%

# Prediction on Train Data
preds = model.predict(X_train)
pd.crosstab(preds, y_train, rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == y_train) # Train Data Accuracy # 97% accuracy 


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, model.predict(X_test)))


import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)
