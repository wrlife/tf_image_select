from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os

class DataLoader(object):
	def __init__(self,
				 dataset_dir,
				 batch_size,
				 image_height,
				 image_width,
				 split):
		self.dataset_dir=dataset_dir
		self.batch_size=batch_size
		self.image_height=image_height
		self.image_width=image_width
		self.split=split


	def load_train_batch(self):

		# Reads pfathes of images together with their labels
		image_list, label_list = self.read_labeled_image_list()

		images = tf.convert_to_tensor(image_list, dtype=tf.string)
		labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

		# Makes an input queue
		input_queue = tf.train.slice_input_producer([images, labels],
													num_epochs = 100,
		                                            shuffle=True)

		image, label = self.read_images_from_disk(input_queue)

		# Optional Preprocessing or Data Augmentation
		# tf.image implements most of the standard image augmentation
		if(self.split=="train"):
			image = self.data_augmentation(image,224,224)
		#image = preprocess_image(image)
		#label = preprocess_label(label)

		# Optional Image and Label Batching
		#mage.set_shape([self.image_height, self.image_width, 3])
		image.set_shape([224, 224, 3])
		image_batch, label_batch = tf.train.batch([image, label],
		                                         batch_size=self.batch_size)
		
		return image_batch,label_batch
	

	def read_labeled_image_list(self):
	    """Reads a .txt file containing pathes and labeles
	    Args:
	       image_list_file: a .txt file with one /path/to/image per line
	       label: optionally, if set label will be pasted after each line
	    Returns:
	       List with all filenames in file image_list_file
	    """
	    f = open(self.dataset_dir+'/'+self.split+'.txt', 'r')
	    filenames = []
	    labels = []
	    for line in f:
	        filename, label = line[:-1].split(' ')
	        filename = self.dataset_dir+'/'+filename
	        filenames.append(filename)
	        labels.append(int(label))
	    return filenames, labels


	def read_images_from_disk(self,input_queue):
		"""Consumes a single filename and label as a ' '-delimited string.
		Args:
		  filename_and_label_tensor: A scalar string tensor.
		Returns:
		  Two tensors: the decoded image, and the string label.
		"""

		label = input_queue[1]

		label = tf.one_hot(label,2)

		image_file = tf.read_file(input_queue[0])

		if(self.split=="validate"):
			image = tf.to_float(tf.image.resize_images(tf.image.decode_jpeg(image_file),[224,224]))
		else:
			image = tf.to_float(tf.image.resize_images(tf.image.decode_jpeg(image_file),[256,256]))


		return image, label


	def data_augmentation(self, im, out_h, out_w):
	        # Random scaling
	       


	        # Random cropping
	        def random_cropping(im, out_h, out_w):
	            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
	            in_h, in_w, _ = tf.unstack(tf.shape(im))
	            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
	            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
	            im = tf.image.crop_to_bounding_box(
	                im, offset_y, offset_x, out_h, out_w)

	            return im


	        im = random_cropping(im, out_h, out_w)

	        im = tf.image.random_flip_left_right(im)

	        im = tf.image.random_flip_up_down(im)

	        #im = tf.per_image_standardization(im)

	        #im = tf.cast(im, dtype=tf.uint8)

	        return im