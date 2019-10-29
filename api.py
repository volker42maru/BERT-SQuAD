from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import QA
import logging
import time
import torch
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import onnx_tf

app = Flask(__name__)
CORS(app)

ONNX_PATH = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.onnx"
# TF_PB_PATH = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.pb"
MODEL_PATH = "/home/volker/workspace/data/spanbert_base_squad2"
# model = QA("/home/volker/workspace/data/spanbert_base_squad2")
# model = QA("roberta-large")
TF_PB_PATH = "/home/volker/workspace/data/bert/mrc_en_spanbert_base_run_squad/graph/model.graph"
ONNX_TF_PB_PATH = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.pb"


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
    vsl = 'exact'
    # test(export_onnx=True)
    max_batch = 4
    min_batch = 0

    # pytorch tests
    test(use_jit=False, fp16=False, onnx_runtime=False, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(use_jit=True, fp16=False, onnx_runtime=False, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(use_jit=False, fp16=True, onnx_runtime=False, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(use_jit=True, fp16=True, onnx_runtime=False, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(use_jit=False, fp16=False, onnx_runtime=True, vsl=vsl, min_batch=min_batch, max_batch=max_batch)

    # tf tests
    test(tf_onnx=True, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(tf_version=True, vsl=vsl, min_batch=min_batch, max_batch=max_batch)
    test(tf_version=True, use_jit=True, vsl='rounded', min_batch=min_batch, max_batch=max_batch)


def test(use_jit=False, fp16=False, onnx_runtime=False, export_onnx=False, tf_onnx=False, tf_version=False, vsl='none', min_batch=0, max_batch=1, num_predicts=300):
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
    # passages = [document1]

    if tf_onnx or tf_version:
        from multiprocessing import Pool

        convert_onnx_to_tf = False
        if tf_onnx and convert_onnx_to_tf:
            onnx_model = onnx.load(ONNX_PATH)
            # prepare tf representation
            tf_exp = onnx_tf.backend.prepare(onnx_model)
            # export the model
            tf_exp.export_graph(ONNX_TF_PB_PATH)

        onnx_pb_graph = tf.Graph()
        with onnx_pb_graph.as_default():
            tf_pb_path = ONNX_TF_PB_PATH if tf_onnx else TF_PB_PATH
            onnx_pb_graph_def = tf.GraphDef()
            with tf.gfile.GFile(tf_pb_path, 'rb') as fid:
                serialized_graph = fid.read()

            onnx_pb_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(onnx_pb_graph_def, name='')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            if use_jit:
                # config.gpu_options.per_process_gpu_memory_fraction = 0.5
                config.log_device_placement = False
                config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            with tf.Session(config=config) as sess:
                # INFERENCE using session.run
                model = QA(MODEL_PATH, use_jit=use_jit, fp16=fp16, onnx=onnx_runtime, sess=sess, vsl=vsl, tf_onnx=tf_onnx)

                print('-- BENCHMARKING: JIT={} | FP16={} | ONNX_RUNTIME={} | '
                      'TF_ONNX_VERSION={} | TF_VERSION={} | EXACT_VSL={} --'
                      .format(use_jit, fp16, onnx_runtime, tf_onnx, tf_version, vsl))
                for passage_batch in range(min_batch, max_batch):
                    passage_batch = pow(3, passage_batch-1)
                    if passage_batch < 1:
                        passages = [document1]
                    else:
                        passages = []
                        for i in range(passage_batch):
                            passages.append(document1)
                            passages.append(document2)
                            passages.append(document3)

                    if max_batch > 2:
                        num_predicts = 50
                    time_taken, rps = measure_inference(model, passages, question, num_predicts)
                    # print('Time taken for test: {} s'.format(time_taken))
                    print('RPS: {}'.format(rps))

                sess.close()

            del model, sess
    else:
        model = QA(MODEL_PATH, use_jit=use_jit, fp16=fp16, onnx=onnx_runtime, export_onnx=export_onnx, vsl=vsl, onnx_path=ONNX_PATH)

        if not export_onnx:
            print('-- BENCHMARKING: JIT={} | FP16={} | ONNX_RUNTIME={} | '
                  'TF_ONNX_VERSION={} | TF_VERSION={} | EXACT_VSL={} --'
                  .format(use_jit, fp16, onnx_runtime, tf_onnx, tf_version, vsl))
            for passage_batch in range(min_batch, max_batch):
                passage_batch = pow(3, passage_batch-1)
                if passage_batch < 1:
                    passages = [document1]
                else:
                    passages = []
                    for i in range(passage_batch):
                        passages.append(document1)
                        passages.append(document2)
                        passages.append(document3)

                if max_batch > 2:
                    num_predicts = 50
                time_taken, rps = measure_inference(model, passages, question, num_predicts)
                # print('Time taken for test: {} s'.format(time_taken))
                print('RPS: {}'.format(rps))
        del model
        torch.cuda.empty_cache()


def measure_inference(model, passages, question, num_predicts):
    import random
    print('Num passages: {}'.format(len(passages)))
    answer = model.predict(passages, question)
    answer = model.predict(passages, question)
    print('Sanity check for prediction: {}'.format([a_i['answer'] for a_i in answer]))
    noise = " lala"
    start_time = time.time()
    for i in range(num_predicts):
        # input = [p + random.randint(0, 20) * noise for p in passages]
        # passages = [p + " some random text" for p in passages]
        answer = model.predict(passages, question)
        # torch.cuda.empty_cache()
    end_time = time.time()
    time_taken = (end_time - start_time)
    rps = num_predicts / time_taken

    return time_taken, rps


if __name__ == "__main__":
    # app.run('0.0.0.0',port=4246)
    benchmark_inference()
