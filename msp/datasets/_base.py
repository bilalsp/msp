"""
Base IO methods for datasets
"""
from os.path import dirname, join

import tensorflow as tf


def load_sample_data() -> tf.data.Dataset:
    """Load sample MSP dataset"""
    module_path = dirname(__file__)
    path = join(module_path, 'data', 'sample_tf_rand_2021')
    return load_data(path)

def load_data(path) -> tf.data.Dataset:
    """Load tensorflow MSP dataset"""
    dataset = tf.data.experimental.load(path)
    return dataset
