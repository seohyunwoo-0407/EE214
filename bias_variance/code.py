import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from numpy.linalg import norm

def sample(sample_size, interval, noise=1.):
    x = np.linspace(*interval, sample_size)
    ground_truth = 2 * np.sin(1.4 * x)
    y = ground_truth + np.random.randn(x.size) * noise
    return x, y, ground_truth

sample_size = 25
interval = (0, 5)

np.random.seed(2020)
x, y, ground_truth = sample(sample_size, interval)

def fit_polynomial(x, y, degree):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x.reshape(-1,1))
    model = LinearRegression()
    model.fit(x_poly, y)
    return model

def evaluate_polynomial(model, x):
    degree = model.coef_.size - 1
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(x.reshape(-1, 1))
    y = model.predict(X)
    return y

def fit_polynomial_reg(x, y, degree, lambda_):
    poly = PolynomialFeatures(degree)
    model = Ridge(alpha=lambda_)
    X = poly.fit_transform(x.reshape(-1, 1))
    model.fit(X, y)
    return model
 
degree = 5
model = fit_polynomial(x, y, degree)
model_reg = fit_polynomial_reg(x, y, degree, 0.1)
p_y = evaluate_polynomial(model, x)
p_y_reg = evaluate_polynomial(model_reg, x)

lambdas=[0.1, 1, 10, 100]

for lambda_ in lambdas:
    model_reg = fit_polynomial_reg(x, y, degree, lambda_)
    p_y_reg = evaluate_polynomial(model_reg, x)
#    plt.plot(x, p_y_reg, label=f'lambda={lambda_}')


sample_size = 25
n_models = 100
linpenals = np.linspace(start=-7, stop=1, num=20) # Regularization parameter (lambda) = e^-7, e^-6, ..., e^1
lambdas = np.power(np.e, linpenals)

x_test, y_test, ground_truth = sample(sample_size, interval, noise=2.)

bias = []
variance = []

for lambda_ in lambdas:
    avg_y = np.zeros(sample_size)
    models = []
    for i in range(n_models):

        x, y, _ = sample(sample_size, interval, noise=1.)
        model = fit_polynomial_reg(x, y, degree, lambda_)

        p_y = evaluate_polynomial(model, x_test)
        avg_y = avg_y + p_y
        models.append(p_y)

    avg_y = avg_y / n_models

    bias_val = np.sum((avg_y - ground_truth)**2)/sample_size
    bias.append(bias_val)
    var_val = 0
    for p_y in models:
        var_val += np.sum((p_y-avg_y)**2)/sample_size
    variance.append(var_val / n_models)

plt.plot(linpenals, bias, label='(bias)$^{2}$')
plt.plot(linpenals, variance, label='variance')
plt.plot(linpenals, np.array(bias) + np.array(variance), label='(bias)$^{2} + $variance')
plt.xlabel('$\ln\lambda$')
plt.legend()
plt.savefig('bias_variance.png')
plt.show()