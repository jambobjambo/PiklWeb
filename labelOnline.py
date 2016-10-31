import tensorflow as tf
from flask import Flask
app = Flask(__name__)
import urllib


@app.route('/')
def home():
		urllib.urlretrieve("https://firebasestorage.googleapis.com/v0/b/piklappdev.appspot.com/o/imagestmp%2Fimage3.jpg?alt=media&token=4a5c08b4-8d48-45f8-b9ab-5b49dbb14e51", "image.jpg")
		# Read in the image_data
		image_data = tf.gfile.FastGFile('image.jpg', 'rb').read()

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
    app.run(host='127.0.0.1', port=8080)
