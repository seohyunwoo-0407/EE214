import numpy as np
import sklearn
import matplotlib.pyplot as plt

#generate synthetic data
np.random.seed(0)
n=100 #size of dataset
X1 = np.random.normal(loc=(5, 10), scale=5, size=(n, 2)) #평균이 (5,10), 표준편차가 5인 정규분포를 따르는 샘플 100개
X2 = np.random.normal(loc=(20, 20), scale=5, size=(n, 2)) #평균이 (20,20), 표준편차가 5인 정규분포를 따르는 샘플 100개

T1=np.ones(n) #라벨링
T2=np.ones(n)*-1

X_train=np.concatenate((X1, X2), axis=0) #train data 생성성
T_train=np.concatenate((T1, T2), axis=0)

print(X_train.shape) #(200,2)
print(T_train.shape) #(200,)

X1_test = np.random.normal(loc=(5, 10), scale=5, size=(n, 2)) # loc and scale indicate mean and variance
X2_test = np.random.normal(loc=(20, 20), scale=5, size=(n, 2)) # X1 and X2 have different mean
T1_test = np.ones(n)
T2_test = np.ones(n) * -1 

X_test = np.concatenate((X1_test, X2_test)) #test data 생성
T_test = np.concatenate((T1_test, T2_test))

# 데이터 저장
np.save('linear_classification/X_train.npy', X_train)
np.save('linear_classification/T_train.npy', T_train)
np.save('linear_classification/X_test.npy', X_test)
np.save('linear_classification/T_test.npy', T_test)

plt.scatter(X_train[T_train==1][:,0], X_train[T_train==1][:,1], c='b', edgecolor='k', label='label:1', s=35)
plt.scatter(X_train[T_train==-1][:,0], X_train[T_train==-1][:,1], c='r', edgecolor='k', label='label:-1', s=35)
plt.grid(True) #격자표시
plt.legend() #범례 표시

axes=plt.gca()

x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
grids = np.c_[xx.ravel(), yy.ravel()]

ab = np.array([-0.4, -0.3])
c = np.array([8])


Z = grids@ab + c
plt.contour(xx, yy, Z.reshape(xx.shape), levels=[0], colors='k')


plt.savefig('linear_classification/train_dataset.png')
plt.show()