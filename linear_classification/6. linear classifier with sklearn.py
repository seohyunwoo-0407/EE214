from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

x_train=np.load('linear_classification/X_train.npy')
t_train=np.load('linear_classification/T_train.npy')
x_test=np.load('linear_classification/X_test.npy')
t_test=np.load('linear_classification/T_test.npy')

model=Perceptron(eta0=0.01, max_iter=1000, n_iter_no_change=500)

model.fit(x_train, t_train)

ab=model.coef_.T
c=model.intercept_

# Visualize
plt.scatter(x_train[t_train==1][:, 0].T, x_train[t_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(x_train[t_train==-1][:, 0].T, x_train[t_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
plt.grid(True)
plt.legend()

axes = plt.gca()
x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30)) # 30 grids for each axis
grids = np.c_[xx.ravel(), yy.ravel()]

Z = grids@ab + c
plt.contour(xx, yy, Z.reshape(xx.shape), levels=[0], colors='k')
plt.contourf(xx, yy, Z.reshape(xx.shape), cmap='RdBu', alpha=0.7)
plt.savefig('linear_classification/linear_classifier_sklearn.png')
plt.show()

Y_train = model.predict(x_train)
Y_test = model.predict(x_test)

#Print Accuracy
print('Training Accuracy :{:.2f}%'.format(100*accuracy_score(t_train, Y_train)))
print('Test Accuracy :{:.2f}%'.format(100*accuracy_score(t_test, Y_test)))

#출력 결과
#Training Accuracy :95.00%
#Test Accuracy :95.00%