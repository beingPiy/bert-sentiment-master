from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import config
from model import BERTClass



app = Flask(__name__)

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    text = str(sentence)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(dim=0).to(config.DEVICE)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(dim=0).to(config.DEVICE)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(dim=0).to(config.DEVICE)


    outputs = MODEL(ids, mask, token_type_ids)
    outputs = np.array(torch.sigmoid(outputs).detach().cpu())
    ones = np.ones((709, 6))
    outputs = outputs*ones
    return outputs


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form("text")

        prediction = sentence_prediction(sentence)
        movie_df = pd.read_csv("/inputs/movies.csv")
        similarity = cosine_similarity(prediction, movie_df[['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']].values)
        movie_df['similarity'] = similarity[0]
        result = movie_df.sort_values(by=['similarity', 'avg_vote', 'year'], ascending=False).head(10)

        m1 = result.loc[0 , 'original_title']
        m2 = result.loc[1 , 'original_title']
        m3 = result.loc[2 , 'original_title']
        m4 = result.loc[3 , 'original_title']
        m5 = result.loc[4 , 'original_title']

    return render_template('results.html', movie_1 = m1, movie_2 = m2, movie_3 = m3, movie_4 = m4, movie_5 = m5 )


if __name__ == "__main__":
    MODEL = BERTClass()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(config.DEVICE)
    MODEL.eval()
    app.run(host="0.0.0.0", port="9999")
