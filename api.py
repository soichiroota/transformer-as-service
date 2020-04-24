import os
import json

import responder
from transformers import pipeline
import numpy as np


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
MODEL = env['MODEL']
FRAMEWORK = env['FRAMEWORK']

api = responder.API(debug=DEBUG)
nlp = pipeline(
    'feature-extraction',
    model=MODEL,
    tokenizer=MODEL,
    framework=FRAMEWORK)


def get_emb(text):
    embs = nlp(text)
    emb_array = np.array(embs)[0]
    return np.mean(emb_array, axis=0).tolist()


@api.route("/")
async def encode(req, resp):
    body = await req.text
    texts = json.loads(body)
    emb_list = [get_emb(text) for text in texts]
    if emb_list:
        resp_dict = dict(data=emb_list, dim=len(emb_list[0]))
        resp.media = resp_dict
    else:
        resp.media = dict(data=list())


if __name__ == "__main__":
    api.run()