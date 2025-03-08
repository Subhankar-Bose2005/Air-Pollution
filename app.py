from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, static_folder='static')

model = joblib.load('model.joblib')

class_names = ['Good', 'Moderate', 'Poor', 'Hazardous']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(x) for x in request.form.getlist('values')]
        
        if len(values) != 8:
            return render_template('predict.html', prediction='Please provide exactly 8 values.')

        input_data = np.array(values).reshape(1, -1)
        prediction_index = model.predict(input_data)[0]
        prediction = class_names[prediction_index]

        return render_template('predict.html', prediction=prediction)
    except ValueError:
        return render_template('predict.html', prediction='Invalid input. Please enter numeric values.')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
