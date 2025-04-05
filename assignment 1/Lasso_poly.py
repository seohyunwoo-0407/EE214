import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold   
from sklearn.linear_model import LinearRegression
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

def fit_polynomial_lasso(x,y,degree, alpha): # fit polynomial model with ridge regularization
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(x)
    model=Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_poly,y)
    return model, poly

degrees = [1,3,5,7, 9]
alphas = [0.01, 0.1, 1, 10]
k_folds=5

# 결과 저장을 위한 딕셔너리
results = {alpha: {'train_mse': [], 'val_mse': []} for alpha in alphas}

kf=KFold(n_splits=k_folds, shuffle=True, random_state=42)

#각 알파와 degree에 대해 train set 내에서 5-fold cross validation 진행 --> evaluate는 validation set으로 진행행
for alpha in alphas:
    print(f"Testing Alpha: {alpha}")
    degree_train_mse=[]
    degree_val_mse=[]
    for degree in degrees:
        fold_train_mse=[]
        fold_val_mse=[]
        print(f"Testing Degree: {degree}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx] 
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model, poly=fit_polynomial_lasso(X_train_fold,y_train_fold,degree, alpha)

            y_pred_train=evaluate_model(model,poly,X_train_fold,y_train_fold)
            y_pred_val=evaluate_model(model,poly,X_val_fold,y_val_fold)

            train_mse=mean_squared_error(y_train_fold,y_pred_train)
            val_mse=mean_squared_error(y_val_fold,y_pred_val)

            fold_train_mse.append(train_mse)
            fold_val_mse.append(val_mse)

        degree_train_mse.append(np.mean(fold_train_mse))
        degree_val_mse.append(np.mean(fold_val_mse))

    results[alpha]['train_mse']=degree_train_mse
    results[alpha]['val_mse'] =degree_val_mse

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

plt.figure(figsize=(12, 6))

for i, alpha in enumerate(alphas):
    # 각 alpha에 대해 다른 색상 사용
    color = plt.cm.Set1(i/len(alphas))
    
    plt.plot(degrees, results[alpha]['train_mse'], '-o', 
             color=color, label=f'Train MSE (α={alpha})')
    plt.plot(degrees, results[alpha]['val_mse'], '--o', 
             color=color, label=f'Validation MSE (α={alpha})')
plt.plot(degrees, poly_train_errors, color='black', label='Polynomial Training MSE')
plt.plot(degrees, poly_test_errors, color='black', linestyle='--', label='Polynomial Test MSE')

plt.xlabel('Model Complexity (Degree/Number of Basis Functions)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('Lasso_cross_validation_alpha.png')
plt.show()

# Ridge 모델의 최적값 찾기
best_lasso_alpha = None
best_lasso_degree = None
best_lasso_mse = float('inf')

# 각 alpha와 degree 조합에 대해 최소 MSE 찾기 (Ridge)
for alpha in alphas:
    for i, degree in enumerate(degrees):
        val_mse = results[alpha]['val_mse'][i]
        if val_mse < best_lasso_mse:
            best_lasso_mse = val_mse
            best_lasso_alpha = alpha
            best_lasso_degree = degree

# 일반 Polynomial 모델의 최적값 찾기
best_poly_degree = None
best_poly_mse = float('inf')

for i, degree in enumerate(degrees):
    test_mse = poly_test_errors[i]
    if test_mse < best_poly_mse:
        best_poly_mse = test_mse
        best_poly_degree = degree

print("\n=== Best Model Parameters ===")
print("\nLasso Regression:")
print(f"Alpha: {best_lasso_alpha}")
print(f"Degree: {best_lasso_degree}")
print(f"Validation MSE: {best_lasso_mse:.4f}")

print("\nPolynomial Regression (No regularization):")
print(f"Degree: {best_poly_degree}")
print(f"Test MSE: {best_poly_mse:.4f}")