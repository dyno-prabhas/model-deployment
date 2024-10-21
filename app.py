from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

@app.route('/')
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    email = request.form.get("content")
    tokenized_email = cv.transform([email])
    predictions = clf.predict(tokenized_email)
    prediction = "SPAM" if predictions == 1 else "NOT SPAM"

    return render_template("index.html", prediction=prediction, email=email)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8000,debug=True)