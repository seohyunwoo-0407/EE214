import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(10)
n=100
X1, X2 = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30)) # 30x30 총 900개 점 2D grid 생성 
X1, X2 = X1.reshape(900,1), X2.reshape(900,1)
X_train = np.concatenate((X1,X2),axis=1)
T_train = np.ones(900)
for idx in range(len(X_train)):
    if X1[idx]**2+X2[idx]**2 < 9+np.random.randn(1)*3:   #원의방정식 형태인거 같음 x^2+y^2=9+random 원의 방정식
        T_train[idx] = -1

#plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='violet', label='label : 1', s=35)
#plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='aqua', label='label : -1', s=35)
#plt.legend(loc='upper right')
#plt.savefig('nonlinear_svm_data.png')
#plt.show()

clf_highC=svm.SVC(kernel='rbf', C=100)
clf_lowC=svm.SVC(kernel='rbf', C=0.01)

clf_highC.fit(X_train, T_train)
clf_lowC.fit(X_train, T_train)

xx, yy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))

Z_lowC = clf_lowC.decision_function(X_train).reshape(xx.shape)
Z_highC = clf_highC.decision_function(X_train).reshape(xx.shape)

plt.figure(figsize=(15,4))

plt.subplot(121)
plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='violet', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='aqua', label='label : -1', s=35)
plt.grid(True)
plt.legend(loc='upper right')
plt.contour(xx, yy, Z_lowC, colors='k', levels=[0], alpha=0.5)
plt.title('Low C')

print(Z_lowC)

plt.subplot(122)
plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='violet', label='label : 1', s=35)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='aqua', label='label : -1', s=35)
plt.grid(True)
plt.legend(loc='upper right')
plt.contour(xx, yy, Z_highC, colors='k', levels=[0], alpha=0.5)
plt.title('High C')

plt.savefig('nonlinear_svm.png')
plt.show()