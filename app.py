import pickle
from flask import Flask, request, jsonify, app, url_for,render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app= Flask(__name__) 

classifier_model = pickle.load(open('classifier.pkl','rb'))
tidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))

@app.route('/')

def home():
    return "Main Function"


@app.route('/predict_api', methods=['POST'])

def predict_api():
    data= ['President Obama is not performing well. He is a terrorist.']
    print(data)
    trans_data = tidf.transform(data)
    output=classifier_model.predict(trans_data)
    return (output[0])

if __name__ =='__main__':
	app.run(debug=True)