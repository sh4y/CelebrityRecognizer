import tensorflow as tf
import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


def argmax(pairs):
    return max(pairs, key=lambda x: x[1])[0]


def argmax_index(values):
    return argmax(enumerate(values))


def classify_face(image_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,
                                name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        # Output
        names = []
        scores = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            names.append(human_string)
            scores.append(score)
            # print('%s (score = %.5f)' % (human_string, score))
        result = "Name: " + names[argmax_index(scores)] + ", Score: " + str(scores[argmax_index(scores)])
    return result


print(classify_face(sys.argv[1]))