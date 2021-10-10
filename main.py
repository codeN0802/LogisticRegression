from flask import Flask, render_template,request
import pickle
from PIL import Image
import  numpy as np


app = Flask(__name__)

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
    a = np.array(img)
    b = np.transpose(a).reshape(a.shape[0], a.shape[1] * a.shape[2])
    c = b.flatten()
    print(c)

    pred = modelx.predict([c])
    if pred == 0 :
        pred = ' Bạn có thể không mắc bệnh tim'
    else:
        pred = ' Bạn có thể đã mắc bệnh tim'


    return render_template('a.html', data=pred)



if __name__ == '__main__':
    app.run(port=3000, debug=True)

