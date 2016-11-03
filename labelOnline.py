import urllib
import tensorflow as tf
from flask import Flask


app = Flask(__name__)


@app.route('/')
def hello():
	urllib.urlretrieve("https://firebasestorage.googleapis.com/v0/b/piklappdev.appspot.com/o/imagestmp%2Fimage3.jpg?alt=media&token=4a5c08b4-8d48-45f8-b9ab-5b49dbb14e51", "image.jpg")
	image_data = tf.gfile.FastGFile('image.jpg', 'rb').read()
	label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
   		return "Hello World"

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)