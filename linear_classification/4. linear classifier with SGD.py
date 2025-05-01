#Train linear classifier(perceptron) with stochastic gradient descent

import numpy as np
import matplotlib.pyplot as plt

X_train = np.load('linear_classification/X_train.npy')
T_train = np.load('linear_classification/T_train.npy')
X_test = np.load('linear_classification/X_test.npy')
T_test = np.load('linear_classification/T_test.npy')

max_iter= 1000
learning_rate= 0.01

np.random.seed(0)
ab=np.random.normal(size=2)
c=np.random.normal(size=1)

for i in range(max_iter):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    T_train_shuffled = T_train[indices]

    for j in range(len(X_train)):
        Pred_train = np.matmul(X_train_shuffled[j], ab) + c
        Y_train = 2 * ((Pred_train) >= 0) - 1

        loss=Pred_train*(Y_train-T_train_shuffled[j])

        grad_a=X_train_shuffled[j][0]*(Y_train-T_train_shuffled[j])
        grad_b=X_train_shuffled[j][1]*(Y_train-T_train_shuffled[j])
        grad_c=Y_train-T_train_shuffled[j]

        ab[0]-=learning_rate*grad_a
        ab[1]-=learning_rate*grad_b
        c-=learning_rate*grad_c
        

Pred_train = X_train@ab + c
Y_train = 2 * ((Pred_train) >= 0) - 1
Pred_acc_train = np.sum((Y_train == T_train)) / len(Y_train)

Pred_test = X_test@ab + c
Y_test = 2 * ((Pred_test) >= 0) - 1
Pred_acc_test = np.sum((Y_test == T_test)) / len(Y_test)

# Visualize
plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
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
plt.savefig('linear_classification/linear_classifier_SGD.png')
plt.show()

#Print Training Accuracy
print('Training Accuracy :{:.2f}%'.format(100*Pred_acc_train))
print('Test Accuracy :{:.2f}%'.format(100*Pred_acc_test))

#출력 결과
#Training Accuracy :93.00%
#Test Accuracy :93.50%