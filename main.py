from flask import Flask, render_template,request
import pickle
from PIL import Image
import  numpy as np


app = Flask(__name__)
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

        # 1. Khai bao w, b
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)

            # Calculate
            # NHÂN MATRIX VỚI VECTOR
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y))

            # cap nhat w,b
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict_proba(self, X):

        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):


        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

modelx = pickle.load(open('hand.pkl','rb'))
# Press the green button in the gutter to run the script.
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('a.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['img']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = Image.open(image_path)
    a = np.array(img)/255
    c = a.flatten()
    print(c)

    pred = modelx.predict([c])
    preds = modelx.predict_proba([c])* 100


    if pred == [1] :
        pred = f'Chúng tôi dự đoán bạn có {preds[0]:.2f} %  bệnh tim => Bạn có thể  đã mắc bệnh tim'
        return render_template('cobenh.html',data=pred)
    else:
        pred = f'Chúng tôi dự đoán bạn có {preds[0]} %  bệnh tim => Bạn có thể khỏe mạnh'
        return render_template('khongcobenh.html', data=pred)








if __name__ == '__main__':
    app.run(port=3000, debug=True)

