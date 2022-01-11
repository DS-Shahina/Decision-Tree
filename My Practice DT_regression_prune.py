import pandas as pd

df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Decision Tree/movies.csv")
df.info()


# Dummy variables
df.head()

# n-1 dummy variables will be created for n categories
# converting non numeric column into numeric (Discrete)
df = pd.get_dummies(df, columns = ["3D_available", "Genre"], drop_first = True) # If i have n categorier then n dummy variables , drop first column, n-1

df.head()
df.info()
# Input and Output Split
predictors = df.loc[:, df.columns!="Collection"] # input(predictors), everthing except "collection" column
type(predictors)

target = df["Collection"] # output
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0) # create 4 parts of your dataset

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score # mean_squared_error- variance, RMSE-Std deviation, r2_score- r square score - goodness of fit, how well the data is getting fit with that model 
# r2_score -  always lie between 0 to 1 , 0 means it's not at all fitting, 1 is fitting perfectly.
# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)


# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())

# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3) # minimum 3 record should be there to split, if 2 record we can't split.
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3) # if the sample has 3 records then consider as leaf node, don't split it further
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)

# test error is high and train error is low so overfitting case
# r2_score(test) is not equal or close to r2_score(train)
#r2_score(train)>r2_score(test)
# Overfitting, we have to keep experiment..!
