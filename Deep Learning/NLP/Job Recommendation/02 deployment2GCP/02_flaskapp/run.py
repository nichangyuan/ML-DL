from flask import Flask, jsonify, request, render_template, url_for
import json
import numpy as np
import pickle
import re


#cleaning
def clean_text(text):
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Keeping only letters
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    # change capital to lowercase
    text = text.lower()
    # Removing additional whitespaces
    text = re.sub(r" +", ' ', text)
    return text

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import EnglishStemmer
def stem_tokenizer(text):
    stemmer = EnglishStemmer(ignore_stopwords=True)
    # Removing the URL links
    words = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", words).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def index():
    return \
    """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    textarea {
        width: 100%;
        padding: 12px 12px;
        margin: 8px 0;
        box-sizing: border-box
        }
    </style>

    </head>
    <body>
    <!--img src="{{ url_for('static', filename='office.jpg') }}" /--><br>
    <img src="/static/office.jpg">
    <br>
    
    <h2>Predict Data Science Job Category from A Job Description</h2>
    
    <form method="POST" action="/result">
      Paste a Job Description from any job board:<br>
      <textarea name="Copied_Text" rows="10" cols=100%>Requirements: Machine learning and Statistics, Natural Language Processing, numpy and pandas, scikit-learn, computer vision</textarea>
      <br><br>
      <input type="submit" value="Submit" style="font-size: 20px; height: 30px; background-color: light-grey">
    </form> 
    </body>
    </html>
    """


@app.route('/result', methods=['POST'])
def result():
    import pandas as pd
    Copied_Text = request.form["""Copied_Text"""]
    test = pd.Series(Copied_Text)
    #nbclassifier.predict_proba(test)
    pred = model.predict_proba(test)
    jd_dict = {0: 'Data Analyst', 1: 'Data Engineer', 2: 'Data Scientist', }
    cat = 0
    proba = -1
    for inx, ele in enumerate(pred[0]):
        if ele > proba:
            cat, proba = inx, ele
    cat = jd_dict[cat].upper()
    proba = str(round(100*proba, 3))
    return \
    """
    <!DOCTYPE html>
    <html>
    <head>
    </head>
    <body>
    <br><br>
    The Data Science Job Category for This Job Description is<br><br>{0} with a probability of {1}% <br><br>
    
    <form action="/">
      <input type="submit" value="Try a new Job Description" style="font-size: 20px; height: 30px; background-color: light-grey">
    </form> 
    </body>
    </html>
    """.format(cat, proba)


@app.route('/scoring', methods=['POST'])
def get_keywords():
    import pandas as pd
    Copied_Text = request.json["""Copied_Text"""]
    test = pd.Series(Copied_Text)
    #nbclassifier.predict_proba(test)
    pred = model.predict_proba(test)
    jd_dict = {0: 'Data Analyst', 1: 'Data Engineer', 2: 'Data Scientist', }
    cat = 0
    proba = -1
    for inx, ele in enumerate(pred[0]):
        if ele > proba:
            cat, proba = inx, ele
    cat = jd_dict[cat].upper()
    proba = str(round(100*proba, 3))
    results = {"proba":proba, "cat":cat}
    return jsonify(results)


if __name__ == "__main__":
    
    app.run(debug=True, host='0.0.0.0', port=5000)  #IP change for GCP deployment
