"""
Copyright IBM Corp. 2021
"""



import logging

import numpy as np

from ibmfl.data.data_handler import DataHandler
from ibmfl.data.data_util import get_reweighing_weights, get_hist_counts
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 42
SENSITIVE_ATTRIBUTE = 'sex'

class AdultSklearnDataHandler(DataHandler):
    """
    Data handler for Adult dataset to train a Logistic Regression Classifier on scikit-learn.
    TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']
            if 'epsilon' in data_config:
                self.epsilon = data_config['epsilon']

        # load dataset
        training_dataset = self.load_dataset()

        # pre-process the data
        self.training_dataset = self.preprocess(training_dataset)
        x_0 = self.training_dataset.iloc[:, :-1]
        y_0 = self.training_dataset.iloc[:, -1]
        x = np.array(x_0)
        y = np.array(y_0)

        self.x_train, self.x_test, self.y_train, self.y_test =\
            train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def get_data(self):
        """
        Returns pre-processed adult training and testing data.

        :return: training and testing data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_dataset(self):
        """
        Loads the training dataset from a given local path.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        try:
            logger.info('Loaded training data from '+ str(self.file_name))
            training_dataset = pd.read_csv(self.file_name,
                                           dtype='category')
        except Exception:
            raise IOError('Unable to load training data from path '
                          'provided in config file: ' +
                          self.file_name)
        return training_dataset

    def get_weight(self):
        """
        Gets pre-processed adult training and testing data, calculates weights for points
        weight = P-expected(sensitive_attribute & class)/P-observed(sensitive_attribute & class)

        :return: weights
        :rtype: `np.array`
        """
        cols = self.get_col_names()
        training_data, (_) = self.get_data()
        return get_reweighing_weights(training_data, SENSITIVE_ATTRIBUTE, cols)


    def get_hist(self):
        """
        Gets pre-processed adult training and testing data, calculates counts for sensitive attribute
        and label

        :return: weights
        :rtype: `np.array`
        """
        e = self.epsilon
        cols = self.get_col_names()
        training_data, (_) = self.get_data()
        return get_hist_counts(training_data, SENSITIVE_ATTRIBUTE, cols, e)

    @staticmethod
    def get_col_names():
        """
        Returns the names of the dataset columns

        :return: column names
        :rtype: `list`
        """
        cols = ['race', 'sex', 'age1', 'age2', 'age3', 'age4', 'age5', 'age6',
                'age7', 'ed6less', 'ed6', 'ed7', 'ed8', 'ed9',
                'ed10', 'ed11', 'ed12', 'ed12more']

        return cols

    @staticmethod
    def get_sa():
        """
        Returns the sensitive attribute

        :return: sensitive attribute
        :rtype: `str`
        """
        return SENSITIVE_ATTRIBUTE

    def preprocess(self, training_data):
        """
        Performs the following preprocessing on adult training and testing data:
        * Drop following features: 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation',
          'relationship', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        * Map 'race', 'sex' and 'class' values to 0/1
            * ' White': 1, ' Amer-Indian-Eskimo': 0, ' Asian-Pac-Islander': 0, ' Black': 0, ' Other': 0
            * ' Male': 1, ' Female': 0
            * Further details in Kamiran, F. and Calders, T. Data preprocessing techniques for classification without discrimination
        * Split 'age' and 'education' columns into multiple columns based on value

        :param training_data: Raw training data
        :type training_data: `pandas.core.frame.DataFrame
        :return: Preprocessed training data
        :rtype: `pandas.core.frame.DataFrame`
        """
        if len(training_data.columns)==15:
            # drop 'fnlwgt' column
            training_data = training_data.drop(
                training_data.columns[2], axis='columns')

        training_data.columns = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                                    'occupation',
                                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                    'native-country',
                                    'class']

        # filter out columns unused in training, and reorder columns
        training_dataset = training_data.loc[:,['race', 'sex', 'age', 'education-num', 'class']]

        # map 'sex' and 'race' feature values based on sensitive attribute privileged/unpriveleged groups
        training_dataset['sex'] = training_dataset['sex'].map({' Female': 0, ' Male': 1})
        training_dataset['race'] = training_dataset['race'].map(
            {' Asian-Pac-Islander': 0, ' Amer-Indian-Eskimo': 0, ' Other': 0, ' Black': 0, ' White': 1})

        # map 'class' values to 0/1 based on positive and negative classification
        training_dataset['class'] = training_dataset['class'].map({' <=50K': 0, ' >50K': 1})

        training_dataset['age'] = training_dataset['age'].astype(int)
        training_dataset['education-num'] = training_dataset['education-num'].astype(int)

        # split age column into category columns
        for i in range(8):
            if i != 0:
                training_dataset['age' + str(i)] = 0

        for index, row in training_dataset.iterrows():
            if row['age'] < 20:
                training_dataset.loc[index, 'age1'] = 1
            elif ((row['age'] < 30) & (row['age'] >= 20)):
                training_dataset.loc[index, 'age2'] = 1
            elif ((row['age'] < 40) & (row['age'] >= 30)):
                training_dataset.loc[index, 'age3'] = 1
            elif ((row['age'] < 50) & (row['age'] >= 40)):
                training_dataset.loc[index, 'age4'] = 1
            elif ((row['age'] < 60) & (row['age'] >= 50)):
                training_dataset.loc[index, 'age5'] = 1
            elif ((row['age'] < 70) & (row['age'] >= 60)):
                training_dataset.loc[index, 'age6'] = 1
            elif row['age'] >= 70:
                training_dataset.loc[index, 'age7'] = 1

        # split age column into multiple columns
        training_dataset['ed6less'] = 0
        for i in range(13):
            if i >= 6:
                training_dataset['ed' + str(i)] = 0
        training_dataset['ed12more'] = 0

        for index, row in training_dataset.iterrows():
            if row['education-num'] < 6:
                training_dataset.loc[index, 'ed6less'] = 1
            elif row['education-num'] == 6:
                training_dataset.loc[index, 'ed6'] = 1
            elif row['education-num'] == 7:
                training_dataset.loc[index, 'ed7'] = 1
            elif row['education-num'] == 8:
                training_dataset.loc[index, 'ed8'] = 1
            elif row['education-num'] == 9:
                training_dataset.loc[index, 'ed9'] = 1
            elif row['education-num'] == 10:
                training_dataset.loc[index, 'ed10'] = 1
            elif row['education-num'] == 11:
                training_dataset.loc[index, 'ed11'] = 1
            elif row['education-num'] == 12:
                training_dataset.loc[index, 'ed12'] = 1
            elif row['education-num'] > 12:
                training_dataset.loc[index, 'ed12more'] = 1

        training_dataset.drop(['age', 'education-num'], axis=1, inplace=True)

        # move class column to be last column
        label = training_dataset['class']
        training_dataset.drop('class', axis=1, inplace=True)
        training_dataset['class'] = label

        return training_dataset
