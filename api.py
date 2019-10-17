from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import QA
import logging

app = Flask(__name__)
CORS(app)

model = QA("/home/volker/workspace/data/spanbert_base_squad2")
# model = QA("roberta-large")


@app.route("/predict",methods=['POST'])
def predict():
    logging.info('receiving request')
    doc = request.json["document"]
    q = request.json["question"]
    try:
        out = model.predict(doc,q)
        logging.info('processed request: {}'.format(out))
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=4246)