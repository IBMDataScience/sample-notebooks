from keras.preprocessing.image import ImageDataGenerator
import logging

import numpy as np
from keras.utils import np_utils

from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_mnist

logger = logging.getLogger(__name__)



class MnistTFDataHandler(DataHandler):
    """
       Data handler for MNIST dataset.
       """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'train_file' in data_config:
                self.train_file_name = data_config['train_file']
            if 'test_file' in data_config:
                self.test_file_name = data_config['test_file']

    def get_data(self, nb_points=500):
        """
        Gets pre-process mnist training and testing data. Because this method
        is for testing it takes as input the number of datapoints, nb_points,
        to be included in the training and testing set.

        :param: nb_points: Number of data points to be included in each set
        :type nb_points: `int`
        :return: training data
        :rtype: `tuple`
        """
        if self.file_name is None:
            (x_train, y_train), (x_test, y_test) = load_mnist()
            # Reduce datapoints to make test faster
            x_train = x_train[:nb_points]
            y_train = y_train[:nb_points]
            x_test = x_test[:nb_points]
            y_test = y_test[:nb_points]
        else:
            try:
                logger.info(
                    'Loaded training data from ' + str(self.file_name))
                data_train = np.load(self.file_name)
                with open("MNIST-pkl/mnist-keras-train.pkl", 'rb') as f:
                    (x_train, y_train)= pickle.load(f)

                with open("MNIST-pkl/mnist-keras-train.pkl", 'rb') as f:
                    (x_test, y_test)= pickle.load(f)
                
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)

        # Add a channels dimension
        import tensorflow as tf
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        return (x_train, y_train), (x_test, y_test)