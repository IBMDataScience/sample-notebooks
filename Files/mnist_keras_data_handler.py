from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import pickle
import numpy as np

from ibm_watson_machine_learning.federated_learning.data_handler import DataHandler

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
        try:
            logger.info(
                'Loaded training data from ' + str(self.train_file_name))
            with open(self.train_file_name, 'rb') as f:
                (self.x_train, self.y_train)= pickle.load(f)
            logger.info(
                'Loaded test data from ' + str(self.test_file_name))
            with open(self.test_file_name, 'rb') as f:
                (self.x_test, self.y_test)= pickle.load(f)
                
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0

            
        except Exception:
            raise IOError('Unable to load training data from path '
                            'provided in config file: ' +
                            self.train_file_name)

        # Add a channels dimension
        import tensorflow as tf
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]

        print('self.x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        return (self.x_train, self.y_train), (self.x_test, self.y_test)
