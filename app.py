# from crypt import methods
# from wsgiref.util import request_uri
from flask import Flask, render_template, url_for, request
import numpy as np
import pickle

model = pickle.load(open('emp.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def emp():
    projects = request.form['projects']
    hours = request.form['hours']
    time = request.form['time'] 
    accident = request.form['accident'] 
    promotion = request.form['promotion'] 
    prev_satisfaction = request.form['prev_satisfaction']
    satisfaction = request.form['satisfaction'] 
    salary = request.form['salary'] 
    department = request.form['department']

    arr = np.array([[projects, hours, time, accident, promotion, satisfaction, prev_satisfaction, salary, department]]).reshape(1, -1)
    pred = model.predict(arr)
    print(pred)
    return render_template('resp.html', data = pred)

if __name__ == "__main__":
    app.run(debug = True)