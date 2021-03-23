import config
import flask
from flask import Flask
from flask import request
from model import BERTClass
import functools
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, AutoConfig, AutoModel


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
    return outputs[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    ################################################################
    # CODE to be used

    # movie_df = pd.read_csv("../input/imdb-analysis/movies.csv")
    # from sklearn.metrics.pairwise import cosine_similarity
    # ones = np.ones((709, 6))
    # outputs = outputs*ones
    # similarity = cosine_similarity(outputs, movie_df[['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']].values)
    # movie_df['similarity'] = similarity[0]
    # movie_df.sort_values(by=['similarity', 'avg_vote', 'year'], ascending=False).head(10)






    # prediction = sentence_prediction(sentence)
    # processing of prediction
    response = {}
    # response["response"] = {
    #     movie list
    # }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTClass()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(config.DEVICE)
    MODEL.eval()
    app.run(host="0.0.0.0", port="9999")
