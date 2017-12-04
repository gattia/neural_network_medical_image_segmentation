"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/keras/')
import backend as K



def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0,0, o_x], 
                            [0, 1,0, o_y], 
                            [0, 0, 1,o_z],
                            [0,0,0,1]])
    reset_matrix = np.array([[1, 0,0, -o_x], 
                            [0, 1,0, -o_y], 
                            [0, 0, 1,-o_z],
                            [0,0,0,1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0., order=0):
    # print(x.shape)
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    # print('affine shape' + str(final_affine_matrix.shape))
    # print('image shape' + str(x.shape))
    # channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
    #                                                      final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
    for channel in range(x.shape[0]):
        x[channel,:,:,:] = ndi.interpolation.affine_transform(x[channel,:,:,:], final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval)

    # x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class Iterator(object):

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg', order=0):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.x = np.asarray(x, dtype=K.floatx())
        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        # if self.x.shape[channels_axis] not in {1, 3, 4}:
        #     raise ValueError('NumpyArrayIterator is set to use the '
        #                      'dimension ordering convention "' + dim_ordering + '" '
        #                      '(channels on axis ' + str(channels_axis) + '), i.e. expected '
        #                      'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
        #                      'However, it was passed an array with shape ' + str(self.x.shape) +
        #                      ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.order = order
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        # if self.image_data_generator.resize != 0:
        #     batch_x = np.zeros(tuple([current_batch_size] + list((self.x.shape*self.image_data_generator.resize)[1:])), dtype=K.floatx())
        # else:
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            batch_x[i] = x
#         if self.save_to_dir:
#             for i in range(current_batch_size):
#                 img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=current_index + i,
#                                                                   hash=np.random.randint(1e4),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y
    

class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
    """

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default', 
                 order=0):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        self.order=order
        if dim_ordering == 'th':
            self.channel_axis = 1
            self.slice_axis = 2
            self.row_axis = 3
            self.col_axis = 4
        if dim_ordering == 'tf':
            self.channel_axis = 4
            self.slice_axis = 1
            self.row_axis = 2
            self.col_axis = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg', order=0):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format, order = order)

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        # 
        # img_slice_axis = self.slice_axis - 1
        # img_row_axis = self.row_axis - 1
        # img_col_axis = self.col_axis - 1
        # img_channel_axis = self.channel_axis - 1

        img_slice_axis = 1
        img_row_axis = 2
        img_col_axis = 3
        img_channel_axis = 0

        ## IMAGE ROTATIONS ##
        if self.rotation_range:
            thetaX = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            # thetaY = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            # thetaZ = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        thetaY = 0
        thetaZ = 0

        # rotation_matrix_z = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0, 0],
        #                         [np.sin(thetaZ), np.cos(thetaZ), 0, 0],
        #                         [0, 0, 1, 0], 
        #                         [0,0,0,1]])

        rotation_matrix_x = np.array([[1,0,0,0],
                                [0,np.cos(thetaX), -np.sin(thetaX), 0],
                                [0,np.sin(thetaX), np.cos(thetaX), 0],
                                [0, 0, 0,1]])

        # rotation_matrix_y = np.array([[np.cos(thetaY),0, -np.sin(thetaY), 0],
        #                         [0,1,0,0],
        #                         [np.sin(thetaY), 0, np.cos(thetaY), 0],
        #                         [0, 0, 0,1]])

        # rotation_matrix_z = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0, 0],
        #                         [np.sin(thetaZ), np.cos(thetaZ), 0, 0],
        #                         [0, 0, 1, 0], 
        #                         [0,0,0,1]])

        # rotation_matrix_x = np.array([[1,0,0,0],
        #                         [0,1, 0, 0],
        #                         [0,0, 1, 0],
        #                         [0, 0, 0,1]])

        # rotation_matrix_y = np.array([[1,0,0,0],
        #                         [0,1, 0, 0],
        #                         [0,0, 1, 0],
        #                         [0, 0, 0,1]])

        # rotation_matrix = np.dot(np.dot(rotation_matrix_z, rotation_matrix_x), rotation_matrix_y)
        rotation_matrix = rotation_matrix_x
        
        ## IMAGE TRANSLATION ##
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0


        ########    BELOW NEEDS ITS OWN VARIABLE self.depth_shift_range     ########
        if self.width_shift_range:
            tz = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_slice_axis]
        else:
            tz = 0

        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]])
        
        ## IMAGE SHEAR ##
        if self.shear_range:
            shearA = np.random.uniform(-intensity, intensity)
            shearB = np.random.uniform(-intensity, intensity)
            shearC = np.random.uniform(-intensity, intensity)
            shearD = np.random.uniform(-intensity, intensity)
            shearE = np.random.uniform(-intensity, intensity)
            shearF = np.random.uniform(-intensity, intensity)
        else:
            shearA = 0
            shearB = 0
            shearC = 0
            shearD = 0
            shearE = 0
            shearF = 0
        shear_matrix = np.array([[1, shearA, shearB, 0],
                                 [shearC, 1, shearD, 0],
                                 [shearE, shearF, 1, 0],
                                 [0, 0, 0, 1]])
        
        ## IMAGE ZOOM ##
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(zoom_range[0], zoom_range[1], 3)
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])
        
        ## CALCUALTE COMBINED TRANSFORMATION MATRIX ##
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        ## APPLY TRANSFORMATION MATRIX ##
        h, w, d = x.shape[img_row_axis], x.shape[img_col_axis], x.shape[img_slice_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w, d)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval, order=self.order)
        
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = np.flip(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = np.flip(x, img_row_axis)

        # if self.resize != 0:
        #     x = ndi.zoom(x, (1,self.resize, self.resize, self.resize), order=self.order)

        return x
