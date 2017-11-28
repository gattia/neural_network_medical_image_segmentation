'''
This assumes that the segmentation_array will have 4 dimensions when you get it:
dimeensions = [different images, x-dimension, y-dimension, z-dimension]

You will add a dimension to this image (in the [1] location) and hot-encode the labels
along this axis. The resulting hot encoded segmentation will have dimensions as outlined
below. 

hot_encoded_segmentation has 5 dimensions: 
dimensions= [diff images, different labels, x-dimension, y-dimension, z-dimension]
'''

import numpy

def hotEncodeSegmentation(segmentation_array, listLabels=[1]):
	outputImageShape = (segmentation_array[0], len(listLabels)+1, segmentation_array[1], segmentation_array[2], segmentation_array[3])
    segmentation_hotEncoded = numpy.zeros(outputImageShape)
    index =0
    for label in listLabels:
    	locationLabel = numpy.where(segmentation_array==label)
    	segmentation_hotEncoded[locationLabel[0], index, locationLabel[1], locationLabel[2], locationLabel[3]]=1
    	index +=1
    background = numpy.where(segmentation_array==0)
    segmentation_hotEncoded[background[0], index, background[1], background[2], background[3]]
    return segmentation_hotEncoded