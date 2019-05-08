import numpy as np
import tensorflow as tf
from common.records.exceptions import FieldInvalidShapeError


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature_unpack(feature):
    return feature.bytes_list.value


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature_unpack(feature):
    return feature.float_list.value


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature_unpack(feature):
    return feature.int64_list.value


def mul(iterable):
    result = 1
    for item in iterable:
        result *= item
    return result


class BaseField(object):
    AVAILABLE_DTYPES = {
        # dtype: (python_type, pack, unpack)
        tf.float32: (float, _float_feature, _float_feature_unpack),
        tf.int64: (int, _int64_feature, _int64_feature_unpack),
        tf.string: (bytes, _bytes_feature, _bytes_feature_unpack),
    }

    def __init__(self, dtype, shape):
        assert dtype in self.AVAILABLE_DTYPES

        self.dtype = dtype
        self.shape = shape
        self.name = None  # would be set in metaclass

    @property
    def descriptor(self):
        return None

    def process_raw(self, value):
        raise NotImplemented


class ScalarField(BaseField):

    def __init__(self, dtype):
        super().__init__(dtype, (1,))

    @property
    def descriptor(self):
        return {
            self.name: tf.FixedLenFeature([1], self.dtype)
        }

    def process_raw(self, value):
        pytype, pack, _ = self.AVAILABLE_DTYPES[self.dtype]
        return {
            self.name: pack([pytype(value)])
        }


class TensorField(BaseField):

    @property
    def descriptor(self):
        return {
            self.name: tf.FixedLenFeature(self.shape, tf.float32),
        }

    def process_raw(self, value):
        if self.shape != value.shape:
            raise FieldInvalidShapeError(
                '%s : %s != %s' % (self.name, self.shape, value.shape)
            )
        new_shape = mul(self.shape)
        new_value = np.reshape(value, new_shape)
        pack = self.AVAILABLE_DTYPES[self.dtype][1]
        return {
            self.name: pack(new_value)
        }
