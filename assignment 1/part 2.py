from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
1. Data Loading and Preprocessing
'''
# 1-1. Load the dataset using pandas
concrete_compressive_strength = fetch_ucirepo(id=165) 

x = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 

X=pd.DataFrame(x)
Y=pd.DataFrame(y)

print(X.shape) #(1030, 8)
print(Y.shape) #(1030, 1)
# 1-2. Handle missing values if present
X = X.dropna()
Y = Y.dropna()

print(X.shape) #(1030, 8) 
print(Y.shape) #(1030, 1)
#Since shape is same before and after dropping missing values, there are no missing values.

# 1-3. Standardize the features
X = (X - X.mean()) / X.std()
print(X)

# 1-4. Split the dataset into training (80%) and testing (20%) sets
def split_dataset(X, Y): # function to split the dataset into training and testing sets
    X_size=X.shape[0]
    Y_size=Y.shape[0]
    X_train, X_test = X.iloc[:int(X_size*0.8)], X.iloc[int(X_size*0.8):]
    Y_train, Y_test = Y.iloc[:int(Y_size*0.8)], Y.iloc[int(Y_size*0.8):]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, y_train, y_test = split_dataset(X, Y)

print(X_train.shape) #(824, 8)
print(X_test.shape) #(206, 8)
print(y_train.shape) #(824, 1)
print(y_test.shape) #(206, 1)

X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(X, Y, test_size=0.2)

print(X_train_sklearn.shape) #(824, 8)
print(X_test_sklearn.shape) #(206, 8)
print(y_train_sklearn.shape) #(824, 1)
print(y_test_sklearn.shape) #(206, 1)

'''
2. Modeling Approaches
'''
# 2-1. Polynomial Regression Model
degrees=np.arange(1,10)
print(degrees)

# 2-2. Gaussian Basic Regression Model
