import numpy as np
import matplotlib.pyplot as plt

X_train = np.load('linear_classification/X_train.npy')
T_train = np.load('linear_classification/T_train.npy')
X_test = np.load('linear_classification/X_test.npy')
T_test = np.load('linear_classification/T_test.npy')

best_acc=0
max_iter=1000
Pred_acc_best = 0
np.random.seed(0)
for i in range(max_iter):
    ab = np.random.normal(size=2) # 두개의 정규분포 난수 생성 
    c = np.random.normal(size=1) #1개의 정규분포 난수 생성
    Pred_train = np.matmul(X_train, ab) + c
    Y_train = 2 * ((Pred_train) >= 0) - 1 #예측값이 0보다 크거나 같으면 1, 작으면 -1을 변환하여 Y_train 생성
    Pred_acc = np.sum((Y_train == T_train)) / len(Y_train)
    if Pred_acc >= Pred_acc_best:
        Pred_acc_best = Pred_acc # Update the Best Parameter Whenever the Best Training Sampple Accuracy Is Updated
        best_ab = ab
        best_c = c

Pred_train = X_train@best_ab + best_c
Y_train = 2 * ((Pred_train) >= 0) - 1 # 예측값
Pred_acc_train=np.sum((Y_train == T_train)) / len(Y_train)

Pred_test = X_test@best_ab + best_c
Y_test = 2 * ((Pred_test) >= 0) - 1 # 예측값
Pred_acc_test=np.sum((Y_test == T_test)) / len(Y_test)



plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
plt.grid(True)
plt.legend()

axes = plt.gca()
x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30)) # 30 grids for each axis
grids = np.c_[xx.ravel(), yy.ravel()]

Z = grids@best_ab + best_c
plt.contour(xx, yy, Z.reshape(xx.shape), levels=[0], colors='k')
plt.contourf(xx, yy, Z.reshape(xx.shape), cmap='RdBu', alpha=0.7)

plt.savefig('linear_classification/random_search.png')
plt.show()

print(f"Best Accuracy: {Pred_acc_best:.2f}")
print(f"Best Parameters: ab={best_ab}, c={best_c}")
print('Training Accuracy :{:.2f}%'.format(100*Pred_acc_train))
print('Test Accuracy :{:.2f}%'.format(100*Pred_acc_test))