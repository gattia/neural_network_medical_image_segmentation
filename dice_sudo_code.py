'''
This assumes that the y_true and y_pred has 5 dimensions: 
dimensions= [diff images, different labels, x-dimension, y-dimension, z-dimension]
'''

import numpy

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = numpy.flatten(y_true)
    y_pred_f = numpy.flatten(y_pred)
    intersection = numpy.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (numpy.sum(y_true_f) + numpy.sum(y_pred_f) + smooth)

def dice_coef_loss_sigmoid(y_true, y_pred):
    dice=0
    for index in range(y_true.shape[1]):
        dice += -dice_coef(y_true [:,index,:,:,:], y_pred[:,index,:,:,:])
    return dice
