'''
This assumes that the outputSegmentation has 5 dimensions: 
dimensions= [diff images, different labels, x-dimension, y-dimension, z-dimension]
'''
import numpy 

def imageSoftmax(outputSegmentation):
    sigmoid = lambda x: 1/(1+numpy.exp(-x))
    sigmoid_matrix = sigmoid(outputSegmentation)
    softmax_matrix = sigmoid_matrix / K.sum(sigmoid_matrix, axis=1)
    return(softmax_matrix)