import tensorflow as tf
import keras.backend as K
import numpy as np
from DepthMetrics import rmse_metric

def overlap(x1, w1, x2, w2):
	l1 = (x1) - w1 / 2
	l2 = (x2) - w2 / 2
	left = tf.where(K.greater(l1, l2), l1, l2)
	r1 = (x1) + w1 / 2
	r2 = (x2) + w2 / 2
	right = tf.where(K.greater(r1, r2), r2, r1)
	result = right - left
	return result

def iou(x_true, y_true, w_true, h_true, x_pred, y_pred, w_pred, h_pred, t, pred_confid_tf):
	# Truth
	x_true = K.expand_dims(x_true, 2)
	y_true = K.expand_dims(y_true, 2)
	w_true = K.expand_dims(w_true, 2)
	h_true = K.expand_dims(h_true, 2)
	# Pred
	x_pred = K.expand_dims(x_pred, 2)
	y_pred = K.expand_dims(y_pred, 2)
	w_pred = K.expand_dims(w_pred, 2)
	h_pred = K.expand_dims(h_pred, 2)
	# Aux
	xoffset = K.expand_dims(tf.convert_to_tensor(np.asarray([0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
															 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
															 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
															 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
															 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7], dtype=np.float32)), 1)
	yoffset = K.expand_dims(tf.convert_to_tensor(np.asarray([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,
															 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,
															 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,
															 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,
															 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4], dtype=np.float32)), 1)
	# Band: if conf > 0.5 then x_pred
	x = tf.where(t, x_pred, K.zeros_like(x_pred))
	y = tf.where(t, y_pred, K.zeros_like(y_pred))
	w = tf.where(t, w_pred, K.zeros_like(w_pred))
	h = tf.where(t, h_pred, K.zeros_like(h_pred))
	# Overlap
	ow = overlap(x + xoffset, w * 8., x_true + xoffset, w_true * 8.)
	oh = overlap(y + yoffset, h * 5., y_true + yoffset, h_true * 5.)
	ow = tf.where(K.greater(ow, 0), ow, K.zeros_like(ow)) # Just positive overlap
	oh = tf.where(K.greater(oh, 0), oh, K.zeros_like(oh)) # just positive overlap
	# Intersection and union
	intersection = ow * oh
	union = w * 8. * h * 5. + w_true * 8. * h_true * 5. - intersection + K.epsilon()
	# iou
	iouall = intersection / union #(#, 80, 1)
	# counter: num. bb where conf > 0.5
	obj_count = K.sum(tf.where(t, K.ones_like(x_true), K.zeros_like(x_true))) # count where true conf = 1.0
	# iou mean
	ave_iou = K.sum(iouall) / (obj_count + 0.0000001)
	# Counters: true positives, false positives
	recall_t = K.greater(iouall, 0.5) # true if iou greater than 0.5
	fid_t = K.greater(pred_confid_tf, 0.3) # true if conf pred greater than 0.3
	recall_count_all = K.sum(tf.where(fid_t, K.ones_like(iouall), K.zeros_like(iouall))) # true_pos + false_pos
	obj_fid_t = tf.logical_and(fid_t, recall_t) # count If pref greater than 0.3 and iou greater than 0.5
	effevtive_iou_count = K.sum(tf.where(obj_fid_t, K.ones_like(iouall), K.zeros_like(iouall))) # true_positive
	# Recall: true positive / (true positives + false negatives)
	recall = effevtive_iou_count / (obj_count + 0.00000001)
	# Precision: true_positives / (true_positive + false_positives)
	precision = effevtive_iou_count / (recall_count_all + 0.0000001)
	return ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h

def iou_metric(y_true, y_pred):
	# Truth tensor: append bb
	truth_conf_tensor = K.expand_dims(K.concatenate([y_true[:, :, 0], y_true[:, :, 7], y_true[:, :, 14], y_true[:, :, 21], y_true[:, :, 28]], axis=1), 2)
	truth_xy_tensor = K.concatenate([y_true[:, :, 1:3], y_true[:, :, 8:10], y_true[:, :, 15:17], y_true[:, :, 22:24], y_true[:, :, 29:31]], axis=1)
	truth_wh_tensor = K.concatenate([y_true[:, :, 3:5], y_true[:, :, 10:12], y_true[:, :, 17:19], y_true[:, :, 24:26], y_true[:, :, 31:33]], axis=1)
	# Pred tensor
	pred_conf_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 0], y_pred[:, :, 7], y_pred[:, :, 14], y_pred[:, :, 21], y_pred[:, :, 28]], axis=1), 2)
	pred_xy_tensor = K.concatenate([y_pred[:, :, 1:3], y_pred[:, :, 8:10], y_pred[:, :, 15:17], y_pred[:, :, 22:24], y_pred[:, :, 29:31]], axis=1)
	pred_wh_tensor = K.concatenate([y_pred[:, :, 3:5], y_pred[:, :, 10:12], y_pred[:, :, 17:19], y_pred[:, :, 24:26], y_pred[:, :, 31:33]], axis=1)
	# Band: is conf greater than 0.5?
	tens = K.greater(truth_conf_tensor, 0.5)
	# call iou
	ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
																						 truth_xy_tensor[:, :, 1],
																						 truth_wh_tensor[:, :, 0],
																						 truth_wh_tensor[:, :, 1],
																						 pred_xy_tensor[:, :, 0],
																						 pred_xy_tensor[:, :, 1],
																						 pred_wh_tensor[:, :, 0],
																						 pred_wh_tensor[:, :, 1],
																						 tens, pred_conf_tensor)
	return ave_iou

def recall(y_true, y_pred):
	# Truth tensor: append bb
	truth_conf_tensor = K.expand_dims(K.concatenate([y_true[:, :, 0], y_true[:, :, 7], y_true[:, :, 14], y_true[:, :, 21], y_true[:, :, 28]], axis=1), 2)
	truth_xy_tensor = K.concatenate([y_true[:, :, 1:3], y_true[:, :, 8:10], y_true[:, :, 15:17], y_true[:, :, 22:24], y_true[:, :, 29:31]], axis=1)
	truth_wh_tensor = K.concatenate([y_true[:, :, 3:5], y_true[:, :, 10:12], y_true[:, :, 17:19], y_true[:, :, 24:26], y_true[:, :, 31:33]], axis=1)
	# Pred tensor
	pred_conf_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 0], y_pred[:, :, 7], y_pred[:, :, 14], y_pred[:, :, 21], y_pred[:, :, 28]], axis=1), 2)
	pred_xy_tensor = K.concatenate([y_pred[:, :, 1:3], y_pred[:, :, 8:10], y_pred[:, :, 15:17], y_pred[:, :, 22:24], y_pred[:, :, 29:31]], axis=1)
	pred_wh_tensor = K.concatenate([y_pred[:, :, 3:5], y_pred[:, :, 10:12], y_pred[:, :, 17:19], y_pred[:, :, 24:26], y_pred[:, :, 31:33]], axis=1)
	# Band: is conf greater than 0.5?
	tens = K.greater(truth_conf_tensor, 0.5)
	# call iou
	ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
																						 truth_xy_tensor[:, :, 1],
																						 truth_wh_tensor[:, :, 0],
																						 truth_wh_tensor[:, :, 1],
																						 pred_xy_tensor[:, :, 0],
																						 pred_xy_tensor[:, :, 1],
																						 pred_wh_tensor[:, :, 0],
																						 pred_wh_tensor[:, :, 1],
																						 tens, pred_conf_tensor)
	return recall

def precision(y_true, y_pred):
	# Truth tensor: append bb
	truth_conf_tensor = K.expand_dims(K.concatenate([y_true[:, :, 0], y_true[:, :, 7], y_true[:, :, 14], y_true[:, :, 21], y_true[:, :, 28]], axis=1), 2)
	truth_xy_tensor = K.concatenate([y_true[:, :, 1:3], y_true[:, :, 8:10], y_true[:, :, 15:17], y_true[:, :, 22:24], y_true[:, :, 29:31]], axis=1)
	truth_wh_tensor = K.concatenate([y_true[:, :, 3:5], y_true[:, :, 10:12], y_true[:, :, 17:19], y_true[:, :, 24:26], y_true[:, :, 31:33]], axis=1)
	# Pred tensor
	pred_conf_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 0], y_pred[:, :, 7], y_pred[:, :, 14], y_pred[:, :, 21], y_pred[:, :, 28]], axis=1), 2)
	pred_xy_tensor = K.concatenate([y_pred[:, :, 1:3], y_pred[:, :, 8:10], y_pred[:, :, 15:17], y_pred[:, :, 22:24], y_pred[:, :, 29:31]], axis=1)
	pred_wh_tensor = K.concatenate([y_pred[:, :, 3:5], y_pred[:, :, 10:12], y_pred[:, :, 17:19], y_pred[:, :, 24:26], y_pred[:, :, 31:33]], axis=1)
	# Band: is conf greater than 0.5?
	tens = K.greater(truth_conf_tensor, 0.5)
	# call iou
	ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
																						 truth_xy_tensor[:, :, 1],
																						 truth_wh_tensor[:, :, 0],
																						 truth_wh_tensor[:, :, 1],
																						 pred_xy_tensor[:, :, 0],
																						 pred_xy_tensor[:, :, 1],
																						 pred_wh_tensor[:, :, 0],
																						 pred_wh_tensor[:, :, 1],
																						 tens, pred_conf_tensor)
	return precision

def mean_metric(y_true, y_pred):
	truth_m_tensor = K.expand_dims(K.concatenate([y_true[:, :, 5], y_true[:, :, 12], y_true[:, :, 19], y_true[:, :, 26], y_true[:, :, 33]], axis=1), 2)
	pred_m_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 5], y_pred[:, :, 12], y_pred[:, :, 19], y_pred[:, :, 26], y_pred[:, :, 33]], axis=1), 2)
	return rmse_metric(truth_m_tensor,pred_m_tensor)

def variance_metric(y_true, y_pred):
	truth_v_tensor = K.expand_dims(K.concatenate([y_true[:, :, 6], y_true[:, :, 13], y_true[:, :, 20], y_true[:, :, 27], y_true[:, :, 34]], axis=1), 2)
	pred_v_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 6], y_pred[:, :, 13], y_pred[:, :, 20], y_pred[:, :, 27], y_pred[:, :, 34]], axis=1), 2)
	return rmse_metric(truth_v_tensor, pred_v_tensor)

def yolo_objconf_loss(y_true, y_pred, t):
	# zero if sig(conf_true) < 0.5
	real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
	pobj = K.sigmoid(y_pred) # sigmoid y_pred
	lo = K.square(real_y_true - pobj)
	loss1 = tf.where(t, lo, K.zeros_like(y_true))
	loss = K.sum(loss1)
	return loss

def yolo_nonobjconf_loss(y_true, y_pred, t):
	# zero if sig(conf_true) < 0.5
	real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
	pobj = K.sigmoid(y_pred) # sigmoid y_pred
	lo = K.square(real_y_true - pobj)
	loss1 = tf.where(t, K.zeros_like(y_true), lo)
	loss = K.sum(loss1)
	return loss

def yolo_xy_loss(y_true, y_pred, t):
	lo = K.square(y_true - y_pred)
	loss1 = tf.where(t, lo, K.zeros_like(y_true))
	return K.sum(loss1)

def yolo_wh_loss(y_true, y_pred, t):
	lo = K.square(y_true - y_pred)
	loss1 = tf.where(t, lo, K.zeros_like(y_true))
	return K.sum(loss1)

def yolo_regressor_loss(y_true, y_pred, t):
	lo = K.square(y_true - y_pred)
	loss1 = tf.where(t, lo, K.zeros_like(y_true))
	return K.sum(loss1)

def yolo_v2_loss(y_true, y_pred):
	# Truth tensor: append bb
	truth_conf_tensor = K.expand_dims(K.concatenate([y_true[:, :, 0], y_true[:, :, 7], y_true[:, :, 14], y_true[:, :, 21], y_true[:, :, 28]], axis=1), 2)
	truth_xy_tensor = K.concatenate([y_true[:, :, 1:3], y_true[:, :, 8:10], y_true[:, :, 15:17], y_true[:, :, 21:23], y_true[:, :, 29:31]], axis=1)
	truth_wh_tensor = K.concatenate([y_true[:, :, 3:5], y_true[:, :, 10:12], y_true[:, :, 17:19], y_true[:, :, 23:25], y_true[:, :, 31:33]], axis=1)
	truth_m_tensor = K.expand_dims(K.concatenate([y_true[:, :, 5], y_true[:, :, 12], y_true[:, :, 19], y_true[:, :, 25], y_true[:, :, 33]], axis=1), 2)
	truth_v_tensor = K.expand_dims(K.concatenate([y_true[:,:,6], y_true[:, :, 13], y_true[:,:,20], y_true[:, :, 26], y_true[:, :, 34]], axis=1), 2)
	# Pred tensor
	pred_conf_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 0], y_pred[:, :, 7], y_pred[:, :, 14], y_pred[:, :, 21], y_pred[:, :, 28]], axis=1), 2)
	pred_xy_tensor = K.concatenate([y_pred[:, :, 1:3], y_pred[:, :, 8:10], y_pred[:, :, 15:17], y_pred[:, :, 22:24], y_pred[:, :, 29:31]], axis=1)
	pred_wh_tensor = K.concatenate([y_pred[:, :, 3:5], y_pred[:, :, 10:12], y_pred[:, :, 17:19], y_pred[:, :, 24:26], y_pred[:, :, 31:33]], axis=1)
	pred_m_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 5], y_pred[:, :, 12], y_pred[:, :, 19], y_pred[:, :, 26], y_pred[:, :, 33]], axis=1), 2)
	pred_v_tensor = K.expand_dims(K.concatenate([y_pred[:, :, 6], y_pred[:, :, 13], y_pred[:, :, 20], y_pred[:, :, 27], y_pred[:, :, 34]], axis=1), 2)
	# true if conf_true > 0.5 
	tens = K.greater(K.sigmoid(truth_conf_tensor), 0.5)
	tens_2d = K.concatenate([tens,tens], axis=-1)
	# yolo loss
	obj_conf_loss = yolo_objconf_loss(truth_conf_tensor, pred_conf_tensor, tens)
	nonobj_conf_loss = yolo_nonobjconf_loss(truth_conf_tensor, pred_conf_tensor, tens)
	xy_loss = yolo_xy_loss(truth_xy_tensor, pred_xy_tensor, tens_2d)
	wh_loss = yolo_wh_loss(truth_wh_tensor, pred_wh_tensor, tens_2d)
	m_loss = yolo_regressor_loss(truth_m_tensor, pred_m_tensor, tens)
	v_loss = yolo_regressor_loss(truth_v_tensor, pred_v_tensor, tens)
	# total
	loss = 5.0 * obj_conf_loss + 0.05 * nonobj_conf_loss + 0.25 * xy_loss + 0.25 * wh_loss + 1.5 * m_loss + 1.25 * v_loss
	return loss