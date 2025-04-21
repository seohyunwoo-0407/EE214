import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris=load_iris()
X=iris.data
t=iris.target

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=0)

model=Perceptron(eta0=0.01, max_iter=1000)

model.fit(X_train, t_train)

Y_pred_train=model.predict(X_train)
Y_pred_test=model.predict(X_test)

print('Training Accuracy :{:.2f}%'.format(100*accuracy_score(t_train, Y_pred_train)))
print('Test Accuracy :{:.2f}%'.format(100*accuracy_score(t_test, Y_pred_test)))







