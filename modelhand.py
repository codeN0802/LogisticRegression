from PIL import Image
import pickle
import  numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import warnings

x_train = []
y_train = []
x_test =[]
y_test=[]

def createFileList(myDir, format='.jpg'):
    fileList = []
    # print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList
mydirtrain = 'stare_train'
mydirtest = 'stare_test'
myFileListTrain = createFileList(mydirtrain)
myFileListTest = createFileList(mydirtest)
for file in myFileListTrain:
    if file.__contains__("0_train"):
        y_train.append(0)
    if file.__contains__("1_train"):
        y_train.append(1)
    img_file = Image.open(file)
    a = np.array(img_file)/255
    c = a.flatten()
    x_train.append(c)
x_train1=np.array(x_train)
y_train1=np.array(y_train)


print("===================")
for file in myFileListTest:
    if file.__contains__("0_test"):
        y_test.append(0)
    if file.__contains__("1_test"):
        y_test.append(1)
    img_file = Image.open(file)
    a = np.array(img_file)/255
    c = a.flatten()
    x_test.append(c)

x_test1 = (np.array(x_test))
y_test1 = (np.array(y_test))


class LogisticRegression:
    def __init__(self, learning_rate=0.0018, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _binary_cross_entropy(y, y_hat):
        def safe_log(x):
            return 0 if x == 0 else np.log(x)

        total = 0
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
        return - total / len(y)

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        #  gradient descent
        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)

            # Calculate
            # NHÂN MATRIX VỚI VECTOR
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y))

            # Cap nhat the w ,b
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict_proba(self, X):

        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):


        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]


warnings.filterwarnings('ignore')
model = LogisticRegression()
model.fit(x_train1, y_train1)
pickle.dump(model, open('hand.pkl', 'wb'))
print(model.weights)
print(model.bias)
preds = model.predict(x_test1)
print(preds)
print(accuracy_score(y_test1, preds))
print("CONFUSION MATRIX CỦA MÔ HÌNH")
print(confusion_matrix(y_test1, preds))

print("Precision:",precision_score(y_test1, preds))
print("Recall:",recall_score(y_test1, preds))