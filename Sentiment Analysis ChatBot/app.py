from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    for i in range(5):
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans("", "", string.punctuation))
        score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
        score_result = f"positive: {score['pos']} negative: {score['neg']} neutral: {score['neu']}"
        if score['pos'] > score['neg'] and score['pos'] > score['neu']:
            result = "positive"
        elif score['neg'] > score['neu']:
            result = "negative"
        else:
            result = "neutral"
        return result


if __name__ == '__main__':
    app.run()


