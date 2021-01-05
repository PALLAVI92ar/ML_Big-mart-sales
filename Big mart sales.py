#Big mart sales prediction
#Developed a product to predict outlet sales
#Data Validation, Data Exploration, classification technique, Decision Trees, Pruning, Bagging 
#Python, numpy, pandas, scikit-learn, matplotlib

# Importing libraries
import pandas as pd
import numpy as np

# reading train and test data
train_data = pd.read_csv('big_mart_Train.csv')
# Dataset dimensions - (rows, columns)
train_data.shape
# Features data-type
train_data.dtypes
# List of features
list(train_data)

test_data = pd.read_csv('big_mart_test.csv')
# Dataset dimensions - (rows, columns)
test_data.shape
# Features data-type
test_data.dtypes
# List of features
list(test_data)

#checking the null values
train_data.isnull().sum()
test_data.isnull().sum()

full_data = [train_data, test_data]

# filling null values
for data in full_data:
    data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)
    data['Outlet_Size'].fillna('Medium',inplace = True)
    
#checking the null values
train_data.isnull().sum()
test_data.isnull().sum()


#applying label encoding for object data variables of train and test data and convert into continuous data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train_data["Item_Identifier"]=LE.fit_transform(train_data["Item_Identifier"])
train_data["Item_Fat_Content"]=LE.fit_transform(train_data["Item_Fat_Content"])
train_data["Item_Type"]=LE.fit_transform(train_data["Item_Type"])
train_data["Outlet_Identifier"]=LE.fit_transform(train_data["Outlet_Identifier"])
train_data["Outlet_Size"]=LE.fit_transform(train_data["Outlet_Size"])
train_data["Outlet_Location_Type"]=LE.fit_transform(train_data["Outlet_Location_Type"])
train_data["Outlet_Type"]=LE.fit_transform(train_data["Outlet_Type"]) 
# Dataset dimensions - (rows, columns)          
train_data.shape
# List of features
list(train_data)
# Features data-type
train_data.dtypes

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
test_data["Item_Identifier"]=LE.fit_transform(test_data["Item_Identifier"])
test_data["Item_Fat_Content"]=LE.fit_transform(test_data["Item_Fat_Content"])
test_data["Item_Type"]=LE.fit_transform(test_data["Item_Type"])
test_data["Outlet_Identifier"]=LE.fit_transform(test_data["Outlet_Identifier"])
test_data["Outlet_Size"]=LE.fit_transform(test_data["Outlet_Size"])
test_data["Outlet_Location_Type"]=LE.fit_transform(test_data["Outlet_Location_Type"])
test_data["Outlet_Type"]=LE.fit_transform(test_data["Outlet_Type"])           
# Dataset dimensions - (rows, columns)
test_data.shape
# List of features
list(test_data)
# Features data-type
test_data.dtypes

#splitting train and test variables into X and Y variables 
X_train=train_data.iloc[:,0:11]
X_train

Y_train=train_data["Item_Outlet_Sales"]
Y_train

X_test=test_data.iloc[:,0:11]
X_test

# Implementing the Decision tree model by gini index method
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(X_train, Y_train)

dt.tree_.node_count
dt.tree_.max_depth

Y_pred_train = dt.predict(X_train)
Y_pred_test = dt.predict(X_test)

print(f"Decision tree has {dt.tree_.node_count} nodes with maximum depth covered up to {dt.tree_.max_depth}")

# Further tuning is required to decide about max depth value ie, by pruning
# apply grid search cv method and pass levels with cv = 10 and look
# out for the best depth at this place
from sklearn.model_selection import GridSearchCV
levels = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]}

dtgrid = GridSearchCV(dt, cv = 10, scoring = 'neg_mean_squared_error', param_grid = levels)

dtgridfit = dtgrid.fit(X_train,Y_train)

dtgridfit.fit(X_test,Y_pred_test)

print(np.sqrt(abs(dtgridfit.best_score_)))

print(dtgridfit.best_estimator_)

# After pruning max_depth=6
from sklearn.tree import DecisionTreeRegressor
dt1 = DecisionTreeRegressor(max_depth=6)

dt1.fit(X_train, Y_train)

dt1.tree_.node_count
dt1.tree_.max_depth

Y_pred_train = dt1.predict(X_train)
Y_pred_test = dt1.predict(X_test)

print(f"Decision tree has {dt1.tree_.node_count} nodes with maximum depth covered up to {dt1.tree_.max_depth}")

#DecisionTreeRegressor= dt--> base learner,, To the baselearner dt apply bagging
from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=dt,max_samples=0.6, n_estimators=500, random_state = 40)
bag.fit(X_train, Y_train)
Y_pred = bag.predict(X_test)
Y_pred

# Grid Search Cv method
from sklearn.model_selection import GridSearchCV
samples = {'max_samples': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

bag_grid = GridSearchCV(bag, cv = 10, scoring = 'neg_mean_squared_error', param_grid = samples)

bag_gridfit = bag_grid.fit(X_train,Y_train)

bag_gridfit.fit(X_test,Y_pred)

print(np.sqrt(abs(bag_gridfit.best_score_)))

print(bag_gridfit.best_estimator_)

