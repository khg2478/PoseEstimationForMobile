import tensorflow as tf
import numpy as np
import os

tfrecord_size = 200

CONFIG = None
TFRECORD_BASE_PATH = ""
TFRECORD_BASE_FILENAME = "pose_estimation"


def set_config(config):
    global CONFIG, TFRECORD_BASE_PATH
    CONFIG = config
    TFRECORD_BASE_PATH = CONFIG['tfrecordpath']

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

writer_train = None
writer_valid = None

def write_2_tfrecord_file(no_img, img_meta_data, is_train) :
	global writer_train, writer_valid, TFRECORD_BASE_PATH, TFRECORD_BASE_FILENAME
	if (is_train) :
		tfrecord_save_path = TFRECORD_BASE_PATH + "/train"
	else :
		tfrecord_save_path = TFRECORD_BASE_PATH + "/valid"
	if not os.path.exists(tfrecord_save_path):
		os.makedirs(tfrecord_save_path)
	img_meta_data = np.array(img_meta_data)
	img = img_meta_data[0]
	heatmap = img_meta_data[1]
	height = img.shape[0]
	width = img.shape[1]
	img_raw = img.tostring()
	heatmap_raw = heatmap.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'image_raw': _bytes_feature(img_raw),
		'heatmap_raw': _bytes_feature(heatmap_raw)}))
	if (is_train) :
		if (writer_train is None) :
			writer_train = tf.python_io.TFRecordWriter(tfrecord_save_path + "/" + TFRECORD_BASE_FILENAME + "_" + str(no_img) + ".tfrecord")
		elif ((no_img % tfrecord_size) == 0) :
			writer_train.close()
			writer_train = tf.python_io.TFRecordWriter(tfrecord_save_path + "/" + TFRECORD_BASE_FILENAME + "_" + str(no_img) + ".tfrecord")
		writer_train.write(example.SerializeToString())
	else :
		if (writer_valid is None) :
			writer_valid = tf.python_io.TFRecordWriter(tfrecord_save_path + "/" + TFRECORD_BASE_FILENAME + "_" + str(no_img) + ".tfrecord")
		elif ((no_img % tfrecord_size) == 0) :
			writer_valid.close()
			writer_valid = tf.python_io.TFRecordWriter(tfrecord_save_path + "/" + TFRECORD_BASE_FILENAME + "_" + str(no_img) + ".tfrecord")
		writer_valid.write(example.SerializeToString())



















