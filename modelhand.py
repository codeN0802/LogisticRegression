from PIL import Image
import pickle
import  numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
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
    a = np.array(img_file)
    # b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
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
    a = np.array(img_file)
    # b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
    c = a.flatten()
    x_test.append(c)

x_test1 = (np.array(x_test))
y_test1 = (np.array(y_test))


class LogisticRegression:
    '''
    A class which implements logistic regression model with gradient descent.
    '''

    def __init__(self, learning_rate=0.00016, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    @staticmethod
    def _sigmoid(x):
        '''
        Private method, used to pass results of the line equation through the sigmoid function.

        :param x: float, prediction made by the line equation
        :return: float
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _binary_cross_entropy(y, y_hat):
        '''
        Private method, used to calculate binary cross entropy value between actual classes
        and predicted probabilities.

        :param y: array, true class labels
        :param y_hat: array, predicted probabilities
        :return: float
        '''

        def safe_log(x):
            return 0 if x == 0 else np.log(x)

        total = 0
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
        return - total / len(y)

    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the logistic regression model.

        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize coefficients
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y))

            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict_proba(self, X):
        '''
        Calculates prediction probabilities for a given threshold using the line equation
        passed through the sigmoid function.

        :param X: array, features
        :return: array, prediction probabilities
        '''
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        '''
        Makes predictions using the line equation passed through the sigmoid function.

        :param X: array, features
        :param threshold: float, classification threshold
        :return: array, predictions
        '''

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
print(confusion_matrix(y_test1, preds))

