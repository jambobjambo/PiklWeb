import tensorflow as tf
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
	# Read in the image_data
	image_data = tf.gfile.FastGFile('testdata/download.jpg', 'rb').read()

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

	# Unpersists graph from file
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)