from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

from imageselect_Dataloader import DataLoader
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 20, "The size of of a sample batch")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")

FLAGS = flags.FLAGS

slim = tf.contrib.slim
resnet_v2 = tf.contrib.slim.nets.resnet_v2

def main(_):

	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
	  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)



	with tf.Graph().as_default():
		
		#tf.logging.set_verbosity(tf.logging.INFO)

		#============================================
		#Load image and label
		#============================================
		imageloader = DataLoader(FLAGS.dataset_dir,
								 FLAGS.batch_size,
								 FLAGS.image_height, 
								 FLAGS.image_width,
								 'train')
		image,label = imageloader.load_train_batch()

		#Load validate image and label
		imageloader_validation = DataLoader(FLAGS.validate_dir,
											FLAGS.batch_size,
											FLAGS.image_height, 
											FLAGS.image_width,
											'validate')
		image_validation,labels_validation = imageloader.load_train_batch()

		#============================================
		#Define the model
		#============================================

		with slim.arg_scope(resnet_v2.resnet_arg_scope()) as scope:
			#Define training network
			predictions, end_points = resnet_v2.resnet_v2_50(image, 
													  num_classes=2, 
													  is_training=True
													  )
			predictions = tf.squeeze(predictions, [1, 2], name='SpatialSqueeze')

			#Define validation network
			predictions_validation,_ = resnet_v2.resnet_v2_50(image_validation, 
													  num_classes=2, 
													  is_training=False,
													  reuse = True
													  )
			predictions_validation = tf.squeeze(predictions_validation, [1, 2], name='SpatialSqueeze')

		    # Restore only the convolutional layers:
			#variables_to_restore = slim.get_variables_to_restore(exclude=['resnet_v2_50/logits/weights','resnet_v2_50/logits/biases'])
			variables_to_restore = slim.get_variables_to_restore()
			checkpoint_path = FLAGS.pretrain_weight_dir
			init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, 
																		 variables_to_restore)


			


			#============================================	
			#Specify the loss function:
			#============================================
			slim.losses.softmax_cross_entropy(predictions, label)
			total_loss = slim.losses.get_total_loss()
			tf.summary.scalar('losses/total_loss', total_loss)

			# Specify the optimization scheme:
			optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,FLAGS.beta1)

			# create_train_op that ensures that when we evaluate it to get the loss,
			# the update_ops are done and the gradient updates are computed.
			train_op = slim.learning.create_train_op(total_loss, optimizer)



			#============================================
			#Validation
			#============================================
			accuracy_validation = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_validation, 1)), tf.to_int32(tf.argmax(labels_validation, 1)))
			tf.summary.scalar('accuracy', accuracy_validation)

			def train_step_fn(session, *args, **kwargs):

				total_loss, should_stop = train_step(session, *args, **kwargs)


				if train_step_fn.step % FLAGS.validation_check == 0:
					accuracy = session.run(train_step_fn.accuracy_validation)
					print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

				train_step_fn.step += 1

				return [total_loss, should_stop]

			train_step_fn.step = 0
			train_step_fn.accuracy_validation = accuracy_validation

			#============================================
			# Load pretrained weights.
			#============================================
			def InitAssignFn(sess):
				sess.run(init_assign_op, init_feed_dict)



			#============================================
			#Start training
			#============================================
			slim.learning.train(train_op, 
								FLAGS.checkpoint_dir,
								save_summaries_secs=20,
								save_interval_secs = 60,
								init_fn=InitAssignFn,
								train_step_fn=train_step_fn
								)		


if __name__ == '__main__':
   tf.app.run()