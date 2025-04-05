from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
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
# don't scailing Y value because it is not a feature
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
# we can also use sklearn.model_selection.train_test_split to split the dataset
#X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(X, Y, test_size=0.2)

'''
2. Modeling Approaches
'''
# 2-1. Polynomial Regression Model
degrees=np.arange(1,10) #degree test from 1 to 9
mse_train_list=[]
mse_test_list=[]

def fit_polynomial(x,y,degree):
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=LinearRegression()
    model.fit(X_poly,y)
    return model, poly

def evaluate_model(model,poly,x,y):
    X_poly=poly.transform(x)
    y_pred=model.predict(X_poly)

    y_np=np.array(y)
    y_pred_np=np.array(y_pred)
    mse=np.mean((y_np-y_pred_np)**2)
    return y_pred, mse
'''
for degree in degrees:
    model, poly=fit_polynomial(X_train,y_train,degree)

    y_pred_train, mse_train=evaluate_model(model,poly,X_train,y_train)
    y_pred_test, mse_test=evaluate_model(model,poly,X_test,y_test)
    
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    print(f"degree: {degree}, mse_train: {mse_train}, mse_test: {mse_test}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train_list, 'b-', label='Training MSE')
plt.plot(degrees, mse_test_list, 'r-', label='Test MSE')
plt.yscale('log')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.savefig('MSE_vs_Polynomial_Degree.png')
plt.show()

def fit_polynomial_Lasso(x, y, degree, alpha):
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_poly,y)
    return model, poly

def fit_polynomial_Ridge(x, y, degree, alpha=1):
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=Ridge(alpha=alpha, max_iter=10000)
    model.fit(X_poly,y)
    return model, poly

for degree in degrees:
    model, poly=fit_polynomial_Lasso(X_train,y_train,degree,alpha=0.01)

    y_pred_train, mse_train=evaluate_model(model,poly,X_train,y_train)
    y_pred_test, mse_test=evaluate_model(model,poly,X_test,y_test)
    
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    print(f"degree: {degree}, mse_train: {mse_train}, mse_test: {mse_test}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train_list, 'b-', label='Training MSE')
plt.plot(degrees, mse_test_list, 'r-', label='Test MSE')
plt.yscale('log')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.savefig('MSE_vs_Polynomial_Degree.png')
plt.show()
'''
# 2-2. Gaussian Basic Regression Model

def fit_gaussian_basic(x, y, degree):
    n_samples, n_features = x.shape
    sigma = 1.0
    
    X_transformed = np.ones((n_samples, (degree + 1) * n_features))
    
    for feature in range(n_features):
        x_feature = x.iloc[:, feature].values
        for j in range(1, degree + 1):
            mu = j
            col_idx = feature * (degree + 1) + j
            X_transformed[:, col_idx] = np.exp(-((x_feature - mu) ** 2) / (2 * sigma ** 2))
    
    model = LinearRegression()
    model.fit(X_transformed, y)
    
    return model, X_transformed

def evaluate_gaussian_basic(model, x, degree):
    n_samples, n_features = x.shape
    sigma = 1.0
    
    X_transformed = np.ones((n_samples, (degree + 1) * n_features))
    
    for feature in range(n_features):
        x_feature = x.iloc[:, feature].values
        
        for j in range(1, degree + 1):
            mu = j
            col_idx = feature * (degree + 1) + j
            X_transformed[:, col_idx] = np.exp(-((x_feature - mu) ** 2) / (2 * sigma ** 2))
    
    return model.predict(X_transformed)

mse_train_list_gaussian=[]
mse_test_list_gaussian=[]

degrees=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for degree in degrees:
    model, X_transformed=fit_gaussian_basic(X_train,y_train,degree)

    y_pred_train=evaluate_gaussian_basic(model,X_train,degree)
    y_pred_test=evaluate_gaussian_basic(model,X_test,degree)

    mse_train=np.mean((np.array(y_train)-y_pred_train)**2)
    mse_test=np.mean((np.array(y_test)-y_pred_test)**2)

    mse_train_list_gaussian.append(mse_train)
    mse_test_list_gaussian.append(mse_test)

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train_list_gaussian, 'b-', label='Training MSE')
plt.plot(degrees, mse_test_list_gaussian, 'r-', label='Test MSE')
plt.yscale('log')
plt.xlabel('Gaussian Basic Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Gaussian Basic Degree')
plt.legend()
plt.grid(True)
plt.savefig('MSE_vs_Gaussian_Basic_Degree.png')
plt.show()


