import pandas as pd
import numpy as np

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns

#perform discretization/bucketing using pd.cut 
# converting Taxable.Income columns into categorical variable
#taxable_income <= 30000 as "Risky" and others are "Good".
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
X = data.drop([' monthly income of employee_fake'], axis=1)
y = data[' monthly income of employee_fake']

# Train Test partition of the data
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42) # n_jobs is basically for parallel processing(parallely the tree have to be grown), n_estimators=500- 500 trees

rf_clf.fit(X_train, y_train)

# Prediction on Test Data
preds = rf_clf.predict(X_test)
pd.crosstab(preds, y_test, rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == y_test) # Test Data Accuracy  - 94%

# Prediction on Train Data
preds = rf_clf.predict(X_train)
pd.crosstab(preds, y_train, rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == y_train) # Train Data Accuracy 97% accuracy 

# it improves model accuracy with train accuracy 97% and test accuracy 94%.

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, rf_clf.predict(X_test)))


