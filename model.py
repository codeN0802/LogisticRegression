import os
from sklearn import metrics
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pickle
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
print(x_train1)
print(y_train1)

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
x_test1 = np.array(x_test)
y_test1 = np.array(y_test)

print(x_test1)
print(y_test1)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train1,y_train1)
pickle.dump(logreg, open('stare.pkl','wb'))
print(logreg.coef_)
print(logreg.intercept_)

# #
y_pred=logreg.predict(x_test1)
cnf_matrix = metrics.confusion_matrix(y_test1, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test1, y_pred))
print("Precision:",metrics.precision_score(y_test1, y_pred))
print("Recall:",metrics.recall_score(y_test1, y_pred))
print('MSE:', metrics.mean_squared_error(y_test1, y_pred))
print(f'Scikit-Learn\'s Final R^2 score: ',logreg.score(x_test1, y_test1))
