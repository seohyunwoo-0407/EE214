import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
1. Data Loading and Preprocessing
'''
# 1-1. Load the dataset using pandas
data = pd.read_excel('Concrete_Data.xls')
print(data.head())
print(data.shape)# (1030, 9)

# 1-2. Handle missing values if present
data=data.dropna()
print(data.shape)# (1030, 9) # no missing values

# 1-3. Standardize the features
X = data.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1).values
y = data['Concrete compressive strength(MPa, megapascals) '].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # don't scailing Y value because it is not a feature

# 1-4. Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

'''
2. Modeling Approaches
'''
#(a) Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def fit_polynomial(x,y,degree): # fit polynomial model
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=LinearRegression()
    model.fit(X_poly,y)
    return model, poly

def evaluate_model(model,poly,x,y): # evaluate polynomial model
    X_poly=poly.transform(x)
    y_pred=model.predict(X_poly)
    return y_pred

degrees = [1,3,5,7,9, 11]
poly_train_errors=[]
poly_test_errors=[]

for degree in degrees:
    model, poly=fit_polynomial(X_train,y_train,degree)
    
    y_pred_train=evaluate_model(model,poly,X_train,y_train)
    y_pred_test=evaluate_model(model,poly,X_test,y_test)
    
    train_mse=mean_squared_error(y_train,y_pred_train)
    test_mse=mean_squared_error(y_test,y_pred_test)

    poly_train_errors.append(train_mse)
    poly_test_errors.append(test_mse)
    print(f"Degree: {degree}, Training MSE: {train_mse}, Testing MSE: {test_mse}")


def fit_polynomial_ridge(x,y,degree): # fit polynomial model with ridge regularization
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=Ridge(alpha=0.1)
    model.fit(X_poly,y)
    return model, poly

def fit_polynomial_lasso(x,y,degree): # fit polynomial model with ridge regularization
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=Lasso(alpha=0.1)
    model.fit(X_poly,y)
    return model, poly

ridge_train_mse=[]
ridge_test_mse=[]
lasso_train_mse=[]
lasso_test_mse=[]


for degree in degrees:
    model, poly=fit_polynomial_ridge(X_train,y_train,degree)

    y_pred_train=evaluate_model(model,poly,X_train,y_train)
    y_pred_test=evaluate_model(model,poly,X_test,y_test)
    
    train_mse=mean_squared_error(y_train,y_pred_train)
    test_mse=mean_squared_error(y_test,y_pred_test)
    
    ridge_train_mse.append(train_mse)
    ridge_test_mse.append(test_mse)
    
    print(f"Degree: {degree}, Training MSE: {train_mse}, Testing MSE: {test_mse}")

for degree in degrees:
    model, poly=fit_polynomial_lasso(X_train,y_train,degree)

    y_pred_train=evaluate_model(model,poly,X_train,y_train)
    y_pred_test=evaluate_model(model,poly,X_test,y_test)

    train_mse=mean_squared_error(y_train,y_pred_train)
    test_mse=mean_squared_error(y_test,y_pred_test)
    
    lasso_train_mse.append(train_mse)
    lasso_test_mse.append(test_mse)
    
    print(f"Degree: {degree}, Training MSE: {train_mse}, Testing MSE: {test_mse}")

#(b) Gaussian Basic Regression Model
def create_gaussian_design_matrix(x, degree, sigma=1.0): # create gaussian design matrix
    n_samples=x.shape[0]
    X_gaussian=np.ones((n_samples, degree+1))
    for j in range(1, degree+1):
        mu=j
        X_gaussian[:,j]=np.exp(-(x-mu)**2/(2*sigma**2))
    return X_gaussian

def fit_gaussian_basis(X,y):
    model_gauss=LinearRegression()
    model_gauss.fit(X,y)
    return model_gauss

x_train_feature = X_train[:, 0]
x_test_feature = X_test[:, 0]
gaussian_degrees = [3, 5, 7, 10, 15]
gauss_train_errors = []
gauss_test_errors = []

for degree in gaussian_degrees:
    X_train_gauss=create_gaussian_design_matrix(x_train_feature,degree, sigma=1.0)
    X_test_gauss=create_gaussian_design_matrix(x_test_feature,degree, sigma=1.0)
    
    model_gauss=fit_gaussian_basis(X_train_gauss,y_train)
    y_pred_train=model_gauss.predict(X_train_gauss)
    y_pred_test=model_gauss.predict(X_test_gauss)
    
    train_mse=mean_squared_error(y_train,y_pred_train)
    test_mse=mean_squared_error(y_test,y_pred_test)
    
    gauss_train_errors.append(train_mse)
    gauss_test_errors.append(test_mse)
    print(f"Degree: {degree}, Training MSE: {train_mse}, Testing MSE: {test_mse}")


plt.figure(figsize=(12, 6))

# Polynomial Regression
plt.plot(degrees, poly_train_errors, color='green', label='Polynomial Training MSE')
plt.plot(degrees, poly_test_errors, color='green', linestyle='--', label='Polynomial Test MSE')

# Gaussian Basis Regression
plt.plot(gaussian_degrees, gauss_train_errors, color='yellow', label='Gaussian Training MSE')
plt.plot(gaussian_degrees, gauss_test_errors, color='yellow', linestyle='--', label='Gaussian Test MSE')

# Ridge Regression
plt.plot(degrees, ridge_train_mse, color='blue', label='Ridge Training MSE')
plt.plot(degrees, ridge_test_mse, color='blue', linestyle='--', label='Ridge Test MSE')

# Lasso Regression
plt.plot(degrees, lasso_train_mse, color='red', label='Lasso Training MSE')
plt.plot(degrees, lasso_test_mse, color='red', linestyle='--', label='Lasso Test MSE')

plt.xlabel('Model Complexity (Degree/Number of Basis Functions)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Model Complexity: Polynomial vs Gaussian')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('model_comparison.png')
plt.show()