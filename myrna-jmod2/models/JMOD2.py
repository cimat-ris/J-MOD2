import os
import sys
import time
import math
import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Convolution2D, Input, Conv2DTranspose
from keras.layers import  Reshape, Flatten, Dropout, Concatenate
from keras.layers.advanced_activations import PReLU,LeakyReLU

from lib.DataAugmentationStrategy import DataAugmentationStrategy
from lib.SampleType import Sample
from lib.DataGenerationStrategy import SampleGenerationStrategy
from lib.Dataset import Dataset
from lib.DepthCallback import PrintBatch, TensorBoardCustom
from lib.DepthMetrics import rmse_metric, logrmse_metric, sc_inv_logrmse_metric
from lib.DepthObjectives import log_normals_loss
from lib.ObstacleDetectionObjectives import yolo_v2_loss, iou_metric, recall, precision, mean_metric, variance_metric

class JMOD2(object):
	def __init__(self, config):
		# tf config
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory_fraction
		tf_config.gpu_options.visible_device_list = "0"
		set_session(tf.Session(config=tf_config))
		# config data
		self.config = config
		self.dataset = {}
		self.training_set = {}
		self.test_set = {}
		self.validation_set = {}
		self.input_w = self.config.input_width
		self.input_h = self.config.input_height
		self.data_augmentation_strategy = DataAugmentationStrategy()
		self.shuffle = True
		# Model
		self.model = self.build_model()
		# Prepare dataset
		self.prepare_data()


	def load_dataset(self):
		dataset = Dataset(self.config, SampleGenerationStrategy(sample_type=Sample))
		dataset_name = 'UnrealDataset'
		return dataset, dataset_name

	def prepare_data(self):
		dataset, dataset_name = self.load_dataset()
		# Read dir
		self.dataset[dataset_name] = dataset
		self.dataset[dataset_name].read_data()
		# Training and test dir
		self.training_set, self.test_set = self.dataset[dataset_name].generate_train_test_data()
		# Get Validation set
		np.random.shuffle(self.training_set)
		train_val_split_idx = int(len(self.training_set)*(1-self.config.validation_split))
		self.validation_set = self.training_set[train_val_split_idx:]
		self.training_set = self.training_set[0:train_val_split_idx]

	def prepare_data_for_model(self, features, label):
		# Normalize input
		features = np.asarray(features)
		features = features.astype('float32')
		features /= 255.0
		# Prepare output : lista de numpy arrays
		labels_depth = np.zeros(shape=(features.shape[0],features.shape[1],features.shape[2],1), dtype=np.float32) # Gray Scale
		labels_obs = np.zeros(shape=(features.shape[0],40,35), dtype=np.float32) # Obstacle output
		i = 0
		for elem in label:
			elem["depth"] = np.asarray(elem["depth"]).astype(np.float32)
			# Why????
			elem["depth"] = -4.586e-09 * (elem["depth"] ** 4) + 3.382e-06 * (elem["depth"] ** 3) - 0.000105 * (elem["depth"] ** 2) + 0.04239 * elem["depth"] + 0.04072
			elem["depth"] /= 39.75
			labels_depth[i,:,:,:] = elem["depth"]
			labels_obs[i,:,:] = np.asarray(elem["obstacles"]).astype(np.float32)
			i +=1
		return features, [labels_depth,labels_obs]

	def train_data_generator(self):
		if self.shuffle:
			np.random.shuffle(self.training_set)
		curr_batch = 0
		self.training_set = list(self.training_set)
		while 1:
			if (curr_batch + 1) * self.config.batch_size > len(self.training_set):
				np.random.shuffle(self.training_set)
				curr_batch = 0
			x_train = []
			y_train = []
			for sample in self.training_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
				# Get input
				features = sample.read_features()
				# Get output
				label = sample.read_labels()
				if self.data_augmentation_strategy is not None:
					features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=False)
				# Append in batch list
				x_train.append(features)
				y_train.append(label)
			x_train, y_train = self.prepare_data_for_model(x_train, y_train)
			curr_batch += 1
			yield x_train , y_train

	def validation_data_generator(self):
		if self.shuffle:
			np.random.shuffle(self.validation_set)
		curr_batch = 0
		while 1:
			if (curr_batch + 1) * self.config.batch_size > len(self.validation_set):
				np.random.shuffle(self.validation_set)
				curr_batch = 0
			x_train = []
			y_train = []
			for sample in self.validation_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
				features = sample.read_features()
				label = sample.read_labels()
				if self.data_augmentation_strategy is not None:
					features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)
				x_train.append(features)
				y_train.append(label)
			x_train, y_train = self.prepare_data_for_model(x_train, y_train)
			curr_batch += 1
			yield x_train , y_train

	def tensorboard_data_generator(self, num_samples):
		# Sample list
		curr_train_sample_list = self.training_set[0:num_samples]
		curr_validation_sample_list = self.validation_set[0: num_samples]
		# Aux
		x_train = []
		y_train = []
		for sample in curr_train_sample_list:
			features = sample.read_features()
			label = sample.read_labels()
			if self.data_augmentation_strategy is not None:
				features, label = self.data_augmentation_strategy.process_sample(features, label)
			x_train.append(features)
			y_train.append(label)
		# Sample validation
		for sample in curr_validation_sample_list:
			features = sample.read_features()
			label = sample.read_labels()
			if self.data_augmentation_strategy is not None:
				features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)
			x_train.append(features)
			y_train.append(label)
		# Prepare
		x_train, y_train = self.prepare_data_for_model(x_train, y_train)
		return x_train , y_train


	def build_depth_model(self):
		# Define input 
		input = Input(shape=(self.config.input_height, self.config.input_width, self.config.input_channel), 
					  name='input')
		# Features red
		vgg19model = VGG19(include_top=False, weights='imagenet', input_tensor=input,
						   input_shape=(self.config.input_height, self.config.input_width, self.config.input_channel))
		# No use last layer
		vgg19model.layers.pop()
		# Last layer output
		output = vgg19model.layers[-1].output
		# Depth part
		x = Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2))(output)
		x = PReLU()(x)
		x = Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2))(x)
		x = PReLU()(x)
		x = Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
		x = PReLU()(x)
		x = Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2))(x)
		x = PReLU()(x)
		out = Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)
		# Depth model
		model = Model(inputs=input, outputs=out)
		return model

	def build_model(self):
		# Depth model
		depth_model = self.build_depth_model()
		#Detection section
		output = depth_model.layers[-10].output
		# Detection layes
		x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv1')(output)
		x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv2')(x)
		x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv3')(x)
		x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv4')(x)
		x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv5')(x)
		x = Convolution2D(1400, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
		x = Reshape((40, 35, 160))(x) # 89600
		x = Convolution2D(160, (3, 3), activation='relu', padding='same', name='det_conv7')(x)
		x = Convolution2D(40, (3, 3), activation='relu', padding='same', name='det_conv8')(x)
		x = Convolution2D(1, (3, 3), activation='linear', padding='same', name='det_conv9')(x)
		# Output detection
		out_detection = Reshape((40, 35), name='detection_output')(x)
		# Model depth and detection
		model = Model(inputs= depth_model.inputs[0], outputs=[depth_model.outputs[0], out_detection])
		# 
		opt = Adam(lr=self.config.learning_rate, clipnorm = 1.)
		model.compile(loss={'depth_output': log_normals_loss, 'detection_output':yolo_v2_loss},
					  optimizer=opt,
					  metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric], 'detection_output': [iou_metric, recall, precision, mean_metric, variance_metric]},
					  loss_weights=[1.0, 1.0])
		model.summary()
		return model

	def train(self, initial_epoch=0):
		# Save model summary
		orig_stdout = sys.stdout
		f = open(os.path.join(self.config.model_dir, 'model_summary.txt'), 'w')
		sys.stdout = f
		print(self.model.summary())

		# Print layers in model summary.txt
		for layer in self.model.layers:
			print(layer.get_config())
		sys.stdout = orig_stdout
		f.close()
		# Save img model summaty
		plot_model(self.model, show_shapes=True, to_file=os.path.join(self.config.model_dir, 'model_structure.png'))
		# Inicial time
		t0 = time.time()
		# Samples per epoch
		samples_per_epoch = int(math.floor(len(self.training_set) / self.config.batch_size))
		# Validation steps
		val_step = int(math.floor(len(self.validation_set) / self.config.batch_size))
		print("Samples per epoch: {}".format(samples_per_epoch))
		# Callbacks
		pb = PrintBatch()
		tb_x, tb_y = self.tensorboard_data_generator(self.config.max_image_summary)
		tb = TensorBoardCustom(self.config, tb_x, tb_y, self.config.tensorboard_dir)
		model_checkpoint = ModelCheckpoint(os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),
										   monitor='loss', verbose=2, save_best_only=False, save_weights_only=False, 
										   mode='auto', period=self.config.log_step) # Save weights every 20 epoch
		# Train
		history = self.model.fit_generator(generator=self.train_data_generator(),
										steps_per_epoch=samples_per_epoch,
										callbacks=[pb, model_checkpoint, tb],
										validation_data=self.validation_data_generator(),
										validation_steps=val_step,
										epochs=self.config.num_epochs,
										verbose=2,
										initial_epoch=initial_epoch)
		# Final time
		t1 = time.time()
		print("Training completed in " + str(t1 - t0) + " seconds")
		return history

	def resume_training(self, weights_file, initial_epoch):
		# Load weights
		self.model.load_weights(weights_file)
		history = self.train(initial_epoch)
		return history
