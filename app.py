from flask import Flask,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)



tfidf_vectorizer = TfidfVectorizer()
@app.route("/")
def home():
    return "RUNNING"
@app.route("/similarity",methods=['POST'])
def predict():
    text1=request.form.get("text1")
    text2=request.form.get("text2")
    corpus=[0,1]
    corpus[0]=text1
    corpus[1]=text2
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return {"similarity score":round(cosine_sim[1][0],2)}
