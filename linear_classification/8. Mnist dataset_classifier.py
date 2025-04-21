import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load the dataset
X_train = np.load('mnist_49/X_train.npy')
X_test = np.load('mnist_49/X_test.npy')
Y_train = np.load('mnist_49/Y_train.npy')
Y_test = np.load('mnist_49/Y_test.npy')

# Basic Descriptions of dataset
print('Shape of data: {}'.format(X_train.shape[1:]))
print('Number of train data: {}'.format(len(Y_train)))
print('Number of test data: {}'.format(len(Y_test)))


# Reshape the training and test examples

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

print('X_train_flatten shape: ' + str(X_train_flatten.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test_flatten shape: ' + str(X_test_flatten.shape))
print('Y_test shape: ' + str(Y_test.shape))

X_train_std = X_train_flatten / 255.
X_test_std = X_test_flatten / 255.

model = Perceptron(eta0=0.1, max_iter=5000)
model.fit(X_train_std, Y_train)
Y_Pred_train = model.predict(X_train_std)
print('Training Accuracy :{:.2f}%'.format(100*accuracy_score(Y_train, Y_Pred_train)))

Y_Pred_test = model.predict(X_test_std)
print('Test Accuracy :{:.2f}%'.format(100*accuracy_score(Y_test, Y_Pred_test)))

# Plotting results
np.random.seed(2020)
idxs = np.random.choice(len(Y_test), 10, replace=False)
label_to_class = {0: '4', 1: '9'}

plt.figure(figsize=(16, 8))
for i, idx in enumerate(idxs):
    plt.subplot(2, 5, i + 1)
    predict = model.predict(X_test_std)
    pred_label = (predict[idx] >= 0.5).astype(int)
    plt.imshow(X_test[idx], cmap='gray_r')
    plt.title('Digit Prediction: {} '.format(label_to_class[pred_label]), fontsize=14)
    plt.xticks([]); plt.yticks([])
plt.show()


#출력 결과
#Training Accuracy :97.49%
#Test Accuracy :96.74%