import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
np.random.seed(10)
n=100
X1 = np.random.normal(loc=(5, 10), scale=2, size=(n, 2)) # loc and scale indicate mean and variance
X2 = np.random.normal(loc=(20, 20), scale=2, size=(n, 2)) # X1 and X2 have different mean
T1 = np.ones(n)
T2 = np.ones(n) * -1                      

# concatenate X1 and X2
X_train = np.concatenate((X1, X2))
T_train = np.concatenate((T1, T2))

X1_test = np.random.normal(loc=(5, 10), scale=2, size=(n, 2)) # loc and scale indicate mean and variance
X2_test = np.random.normal(loc=(20, 20), scale=2, size=(n, 2)) # X1 and X2 have different mean
T1_test = np.ones(n)
T2_test = np.ones(n) * -1                      # labeling as 1 for X1 and -1 for X2

# concatenate X1 and X2
X_test = np.concatenate((X1_test, X2_test))
T_test = np.concatenate((T1_test, T2_test))

outlier_x1, outlier_x2 = np.array([[20,12.5]]), np.array([[5,17.5]])
outlier_t1, outlier_t2 = np.array([1]), np.array([-1])

X_train = np.concatenate((X1, X2, outlier_x1, outlier_x2))
T_train = np.concatenate((T1, T2, outlier_t1, outlier_t2))

X_train_min=np.min(X_train, axis=0)
X_train_max=np.max(X_train, axis=0)
X_train_mms = (X_train - X_train_min) / (X_train_max - X_train_min)      # X_train_mms: preprocessed dataset with 'min-max scaler'

X_train_mean=np.mean(X_train, axis=0)
X_train_std=np.std(X_train, axis=0)

X_train_ss =  (X_train-X_train_mean) / X_train_std           # X_train_ss: preprocessed dataset with 'standard scaler'

plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
plt.grid(True)
plt.legend()

axes = plt.gca()
x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30)) # 30 grids for each axis
grids = np.c_[xx.ravel(), yy.ravel()]

clf_lowC = svm.SVC(kernel='linear', C= 0.01)      ## We use support vector classifer (SVC) in the Scikit-Learn's svm library: We use linear kernel, regularazation parameter is set to C=0.01
clf_highC = svm.SVC(kernel='linear', C= 100)       ## We use support vector classifer (SVC) in the Scikit-Learn's svm library: We use linear kernel, regularazation parameter is set to C=100
############################################################################################

############ Train the above two models with fit method ############
clf_lowC.fit(X_train, T_train)                                                  ## Train clf_lowC
clf_highC.fit(X_train, T_train)                                                 ## Train clf_highC
####################################################################
xy=np.vstack([xx.ravel(), yy.ravel()]).T
Z_lowC = clf_lowC.decision_function(xy).reshape(xx.shape)
Z_highC = clf_highC.decision_function(xy).reshape(xx.shape)

plt.figure(figsize=(15,4))

plt.subplot(121)
plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
plt.grid(True)
plt.legend()
plt.contour(xx, yy, Z_lowC, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('Low C')



plt.subplot(122)
plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='b', edgecolor='k', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='r', edgecolor='k', label='label : -1', s=35)
plt.grid(True)
plt.legend()
plt.contour(xx, yy, Z_highC, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('High C')
  

plt.show()