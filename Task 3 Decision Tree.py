import pandas as pd
import numpy as np

data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Decision Tree/Fraud_check.csv")

data.isnull().sum()
data.dropna()
data.columns

#perform discretization/bucketing using pd.cut 
# converting Taxable.Income columns into categorical variable
#taxable_income <= 30000 as "Risky" and others are "Good".
data["Taxable.Income"] = pd.cut(data["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])

data.info()

#converting into binary , converting non numeric data into categorical(binary) data using get dummies
data = pd.get_dummies(data,columns = ["Taxable.Income"],drop_first=True)
#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
data=pd.get_dummies(data,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)

data.info()

#data["default"]=lb.fit_transform(data["default"])

data["Taxable.Income_Good"].unique()
data['Taxable.Income_Good'].value_counts()

#split input and output
predictors = data.drop(['Taxable.Income_Good'], axis=1)
target = data['Taxable.Income_Good']

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=1) # 70% training and 30% test

from sklearn.tree import DecisionTreeClassifier as DT # for numeric output DecisionTreeRegressor

help(DT)
model = DT(criterion = 'entropy')
model.fit(X_train, y_train)

# Prediction on Test Data
preds = model.predict(X_test)
pd.crosstab(preds, y_test, rownames=['Predictions'], colnames=['Actual'])

np.mean(preds == y_test) # Test Data Accuracy  - 65%

# Prediction on Train Data
preds = model.predict(X_train)
pd.crosstab(preds, y_train, rownames = ['Predictions'], colnames = ['Actual'])

np.mean(preds == y_train) # Train Data Accuracy # 100% accuracy - overfitting case

# overfitting case because train accuracy is high as compared to test accuracy

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, model.predict(X_test)))

