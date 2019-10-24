from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import QA
import logging
import time
import torch
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

app = Flask(__name__)
CORS(app)

ONNX_PATH = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.onnx"
# TF_PB_PATH = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.pb"
MODEL_PATH = "/home/volker/workspace/data/spanbert_base_squad2"
# model = QA("/home/volker/workspace/data/spanbert_base_squad2")
# model = QA("roberta-large")
TF_PB_PATH = "/home/volker/workspace/data/bert/mrc_en_spanbert_base_run_squad/graph/model.graph"


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


def benchmark_inference():
    vsl = False
    # test(export_onnx=True)

    # pytorch tests
    test(xla=False, fp16=False, onnx_runtime=False, vsl=vsl)
    test(xla=True, fp16=False, onnx_runtime=False, vsl=vsl)
    test(xla=False, fp16=True, onnx_runtime=False, vsl=vsl)
    test(xla=False, fp16=False, onnx_runtime=True, vsl=vsl)

    # tf tests
    test(tf_onnx=True, vsl=vsl)


def test(xla=False, fp16=False, onnx_runtime=False, export_onnx=False, tf_onnx=False, vsl=False, num_predicts=300):
    document1 = 'Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. ' \
               'The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle ' \
               'main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters ' \
               'were reused after several months of refitting work for each launch. The external tank was ' \
               'discarded after each flight. and the two solid rocket boosters were reused after several ' \
               'months of refitting work for each launch. The external tank was discarded after each flight.'
    document2 = 'This contrasts with expendable launch systems, where each launch vehicle is launched once ' \
                'and then discarded. No completely reusable orbital launch system has ever been created.'
    document3 = 'A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is ' \
                'capable of launching a payload into space more than once. This contrasts with expendable ' \
                'launch systems, where each launch vehicle is launched once and then discarded. No completely ' \
                'reusable orbital launch system has ever been created.'
    question = 'How many partially reusable launch systems were developed?'
    # passages = [document1, document2, document3, document1, document2, document3, document1, document2, document3]
    # passages = [document1, document2, document3]
    passages = [document1]

    if tf_onnx:
        # onnx_model = onnx.load(ONNX_PATH)
        # tf_exp = prepare(onnx_model)  # prepare tf representation
        # tf_exp.export_graph(TF_PB_PATH)  # export the model

        onnx_pb_graph = tf.Graph()
        with onnx_pb_graph.as_default():
            onnx_pb_graph_def = tf.GraphDef()
            with tf.gfile.GFile(TF_PB_PATH, 'rb') as fid:
                serialized_graph = fid.read()

            onnx_pb_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(onnx_pb_graph_def, name='')

            with tf.Session() as sess:
                model = QA(MODEL_PATH, xla=xla, fp16=fp16, onnx=onnx_runtime, sess=sess, vsl=vsl)

                print('-- BENCHMARKING (passages={}): JIT={} | FP16={} | ONNX_RUNTIME={} | TF_VERSION={} | EXACT_VSL={} --'
                      .format(len(passages), xla, fp16, onnx_runtime, tf_onnx, vsl))
                time_taken, rps = measure_inference(model, passages, question, num_predicts)
                # print('Time taken for test: {} s'.format(time_taken))
                print('RPS: {}'.format(rps))
    else:
        model = QA(MODEL_PATH, xla=xla, fp16=fp16, onnx=onnx_runtime, export_onnx=export_onnx, vsl=vsl, onnx_path=ONNX_PATH)

        if not export_onnx:
            print('-- BENCHMARKING (passages={}): JIT={} | FP16={} | ONNX_RUNTIME={} | TF_VERSION={} | EXACT_VSL={} --'
                  .format(len(passages), xla, fp16, onnx_runtime, tf_onnx, vsl))
            time_taken, rps = measure_inference(model, passages, question, num_predicts)
            # print('Time taken for test: {} s'.format(time_taken))
            print('RPS: {}'.format(rps))
        del model
        torch.cuda.empty_cache()


def measure_inference(model, passages, question, num_predicts):
    start_time = time.time()
    for i in range(num_predicts):
        answer = model.predict(passages, question)
        if i == 0:
            print('Sanity check for prediction: {}'.format([a_i['answer'] for a_i in answer]))
    end_time = time.time()
    time_taken = (end_time - start_time)
    rps = num_predicts / time_taken

    return time_taken, rps


if __name__ == "__main__":
    # app.run('0.0.0.0',port=4246)
    benchmark_inference()
