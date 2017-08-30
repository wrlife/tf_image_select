from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import PIL.Image as pil
from glob import glob


import tensorflow.contrib.slim.nets

from imageselect_Dataloader import DataLoader
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")

FLAGS = flags.FLAGS

slim = tf.contrib.slim
resnet_v2 = tf.contrib.slim.nets.resnet_v2

def main(_):




	with tf.Graph().as_default():
		#Load image and label
		x = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

		img_list = sorted(glob(FLAGS.dataset_dir + '/*.jpg'))



		# # Define the model:
		with slim.arg_scope(resnet_v2.resnet_arg_scope()):

			predictions, endpoint = resnet_v2.resnet_v2_50(x,
		                           num_classes=2,
		                           is_training=False)
		                           #spatial_squeeze=False)

			saver = tf.train.Saver([var for var in tf.model_variables()]) 

			ckpt_file = '/home/wrlife/project/deeplearning/tf_image_select/checkpoints/model.ckpt-240741'
			with tf.Session() as sess:

				saver.restore(sess, ckpt_file)

				for i in range(len(img_list)):

					fh = open(img_list[i],'r')
					I = pil.open(fh)
					I = I.resize((224,224),pil.ANTIALIAS)
					I = np.array(I)

					#import pdb;pdb.set_trace()
					classification = sess.run(predictions,feed_dict={x:I[None,:,:,:]})
					classification = np.squeeze(classification, [0, 1, 2])

					if classification[0]>classification[1]:
						print("The %dth frame is bad" % (i))
						os.system("cp "+img_list[i]+' '+FLAGS.output_dir+"/classifiedbad/")
					else:
						print("The %dth frame is good" % (i))
						os.system("cp "+img_list[i]+' '+FLAGS.output_dir+"/classifiedgood/")



if __name__ == '__main__':
   tf.app.run()