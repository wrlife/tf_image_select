from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os

class DataLoader(object):
	def __init__(self,)

	# Make a queue of file names including all the JPEG images files in the relative
	# image directory.
	filename_queue = tf.train.string_input_producer(
	    tf.train.match_filenames_once("./images/*.jpg"))

	# Read an entire image file which is required since they're JPEGs, if the images
	# are too large they could be split in advance to smaller files or use the Fixed
	# reader to split up the file.
	image_reader = tf.WholeFileReader()

	# Read a whole file from the queue, the first returned value in the tuple is the
	# filename which we are ignoring.
	_, image_file = image_reader.read(filename_queue)

	# Decode the image as a JPEG file, this will turn it into a Tensor which we can
	# then use in training.
	image = tf.image.decode_jpeg(image_file)