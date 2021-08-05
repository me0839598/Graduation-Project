import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
clf_checkpoint= joblib.load('GaussianNB.joblib')
reloaded_vect = clf_checkpoint['preprocessing']
clf_model = clf_checkpoint['model']


em = ["anger", "anticipation", "disgust", "fear", "joy","love", "optimism","pessimism", "sadness", "surprise", "trust"]

def labell_classes(ls):
    arr = []
    for i in range(len(ls)):
        if ls[i] == 1:
            arr.append(em[i])
    return arr

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    feature=features[:-1]
    x=reloaded_vect.transform(feature)
    #final_features = [np.array(int_features)]
    prediction =clf_model.predict(x)
    pred=prediction.toarray()[0]
    classes=labell_classes(pred)
    """
    if prediction == 0:
        output = 'Not Diabetic Patient'
    else:
        output = 'Diabetic Patient'
    """
    return render_template('index.html', prediction_text='The model prediction is {}'.format(classes))


if __name__ == "__main__":
    app.run(debug=True)
