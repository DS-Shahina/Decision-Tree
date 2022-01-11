import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/Diabetes.csv")

# Dummy variables
data.head()
data.info()

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

# Train Test partition of the data
from sklearn.model_selection import train_test_split

#split input and output
X=data[[' Number of times pregnant', ' Plasma glucose concentration', ' Diastolic blood pressure', ' Triceps skin fold thickness',' 2-Hour serum insulin',' Body mass index',' Diabetes pedigree function',' Age (years)']]  # Features
y=data[' Class variable']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42) # n_jobs is basically for parallel processing(parallely the tree have to be grown), n_estimators=500- 500 trees

rf_clf.fit(X_train, y_train)

# Prediction on Test Data
preds = rf_clf.predict(X_test)
pd.crosstab(preds, y_test, rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == y_test) # Test Data Accuracy  - 76%

# Prediction on Train Data
preds = rf_clf.predict(X_train)
pd.crosstab(preds, y_train, rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == y_train) # Train Data Accuracy # 100% accuracy - overfitting case

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, rf_clf.predict(X_test)))

#Showing overfitting,need to optimize
forest_new = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')  # n_estimators is the number of decision trees
forest_new.fit(X_train, y_train)

# Prediction on Test Data
preds = forest_new.predict(X_test)
pd.crosstab(preds, y_test, rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == y_test) # Test Data Accuracy  - 72%

# Prediction on Train Data
preds = forest_new.predict(X_train)
pd.crosstab(preds, y_train, rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == y_train) # Train Data Accuracy # 84% accuracy 



