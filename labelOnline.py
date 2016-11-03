import tensorflow as tf
from flask import Flask


app = Flask(__name__)


@app.route('/')
def hello():
	image_data = tf.gfile.FastGFile('image.jpg', 'rb').read()
	label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			predictions = sess.run(softmax_tensor, \
   				{'DecodeJpeg/contents:0': image_data})

		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			return("{ '%s' : { 'score' : '%.5f' } }" % (human_string, score))

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)