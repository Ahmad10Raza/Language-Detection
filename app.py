from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv=CountVectorizer()

app = Flask(__name__)


model=pickle.load(open('model.pikle','rb'))

## Route for homepage
@app.route('/')
def index():
    return "Hellow World!"


# def predict():
#     text=input("Enter The Text: ")
#     data = cv.transform([text]).toarray()
    
#     prediction=model.predict(data)
#     put_text("The Language is: ",prediction)



def predict():
    text=input("Enter The Text: ")
    data = cv.transform([[text]]).toarray()
    #data=data.toarray()
    #data=vectorizer.transform(text)
    data = data.reshape(1, -1)
    prediction=model.predict(text)
    put_text("The Language is: ",prediction)


app.add_url_rule('/predict','webio_view',webio_view(predict),
                 methods=['GET','POST','OPTIONS'])




if __name__=="__main__":
    app.run(host="0.0.0.0")