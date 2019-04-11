import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class RecordColumn(object):

    def __init__(self):
        self.name = None

        self.raw = None
        self.tensor = None
        self.serialized = None

    @property
    def descriptor(self):
        return None

    def process_raw(self, value):
        raise NotImplemented

    def process_record(self, record):
        raise NotImplemented

    def get_seralized(self):
        if not self.serialized:
            raise RuntimeError('Value has not been serialized')
        return self.serialized

    def get_tensor(self):
        if not self.tensor:
            raise RuntimeError('Value has not been deserialized')
        return self.tensor


class BytesScalarColumn(RecordColumn):

    @property
    def descriptor(self):
        return {
            self.name: tf.FixedLenFeature([], tf.string, default_value='')
        }

    def process_raw(self, value):
        self.raw = bytes(value)
        self.serialized = {
            self.name: _bytes_feature(self.raw)
        }

    def process_record(self, record):
        self.serialized = {
            self.name: record[self.name]
        }
        self.tensor = record[self.name]


class FloatScalarColumn(RecordColumn):

    def __init__(self):
        super().__init__()

    @property
    def descriptor(self):
        return {
            self.name: tf.FixedLenFeature([], tf.float32, default_value=0.0)
        }

    def process_raw(self, value):
        self.raw = float(value)
        self.serialized = {
            self.name: _float_feature(value)
        }

    def process_record(self, record):
        self.serialized = {
            self.name: record[self.name]
        }
        self.tensor = tf.reshape(tf.cast(record[self.name], tf.float32), (1,))


class Tensor1DColumn(RecordColumn):
    XSHAPE = '{}_x_shape'

    def __init__(self):
        super().__init__()
        self.x_shape = None

    @property
    def descriptor(self):
        return {
            self.name: tf.VarLenFeature(tf.float32),
            self.XSHAPE.format(self.name): tf.FixedLenFeature([], tf.int64, default_value=0)
        }

    def process_raw(self, value):
        if len(value.shape) != 1:
            raise RuntimeError('Invalid data shape')

        self.raw = value.tolist()
        self.x_shape = len(self.raw)
        self.serialized = {
            self.name: _float_feature(self.raw),
            self.XSHAPE.format(self.name): _int64_feature(self.x_shape)
        }

    def process_record(self, record):
        x_field = self.XSHAPE.format(self.name)
        self.serialized = {
            self.name: record[self.name],
            x_field: record[x_field]
        }
        shape = tf.parallel_stack([tf.cast(record[x_field], tf.int32)])
        dense = tf.sparse_tensor_to_dense(record[self.name], default_value=0)
        self.tensor = tf.reshape(dense, shape)


class Tensor2DColumn(RecordColumn):
    XSHAPE = '{}_x_shape'
    YSHAPE = '{}_y_shape'

    def __init__(self):
        super().__init__()
        self.x_shape = None
        self.y_shape = None

    @property
    def descriptor(self):
        return {
            self.name: tf.VarLenFeature(tf.float32),
            self.XSHAPE.format(self.name): tf.FixedLenFeature([], tf.int64, default_value=0),
            self.YSHAPE.format(self.name): tf.FixedLenFeature([], tf.int64, default_value=0)
        }

    def process_raw(self, value):
        if len(value.shape) != 2:
            raise RuntimeError('Invalid data shape')

        self.x_shape = value.shape[0]
        self.y_shape = value.shape[1]
        self.raw = np.reshape(value, [self.x_shape * self.y_shape]).tolist()
        self.serialized = {
            self.name: _float_feature(self.raw),
            self.XSHAPE.format(self.name): _int64_feature(self.x_shape),
            self.YSHAPE.format(self.name): _int64_feature(self.y_shape)
        }

    def process_record(self, record):
        x_field = self.XSHAPE.format(self.name)
        y_field = self.YSHAPE.format(self.name)
        self.serialized = {
            self.name: record[self.name],
            x_field: record[x_field],
            y_field: record[y_field]
        }
        shape = tf.parallel_stack([tf.cast(record[x_field], tf.int32), tf.cast(record[y_field], tf.int32)])
        dense = tf.sparse_tensor_to_dense(record[self.name], default_value=0)
        self.tensor = tf.reshape(dense, shape)


class RecordMeta(type):
    DF_ATTR_NAME = 'DATA_FIELDS'

    def __new__(mcs, name, bases, attrs):
        if mcs.DF_ATTR_NAME not in attrs:
            attrs[mcs.DF_ATTR_NAME] = set()

        for attr_name, attr in attrs.items():
            if not isinstance(attr, RecordColumn):
                continue
            attrs[mcs.DF_ATTR_NAME].add(attr_name)
            attr.name = attr_name

        return super().__new__(mcs, name, bases, attrs)


class BaseDataRecord(object, metaclass=RecordMeta):
    DATA_FIELDS = set()
    FEATURES = set()

    def __setitem__(self, key, value):
        if key in self.DATA_FIELDS:
            field = getattr(self, key)
            field.process_raw(value)
        else:
            setattr(self, key, value)

    def __getitem__(self, key):
        if key in self.DATA_FIELDS:
            field = getattr(self, key)
            if field.tensor:
                return field.tensor
            else:
                return field.raw
        else:
            return getattr(self, key)

    def _collect_data(self):
        data = {}
        for item in self.DATA_FIELDS:
            field = getattr(self, item)
            data.update(field.serialized)

        return data

    def serialize(self):
        data = self._collect_data()
        record = tf.train.Features(feature=data)
        example = tf.train.Example(features=record)
        return example.SerializeToString()

    @classmethod
    def deserialize(cls, data_record):
        record_description = {}
        for item in cls.DATA_FIELDS:
            field = getattr(cls, item)
            record_description.update(field.descriptor)

        sample = tf.parse_single_example(data_record, record_description)
        return sample

    @classmethod
    def decompose_on_feature_labels(cls, data_record):
        features, labels = {}, {}
        for item in cls.FEATURES:
            field = getattr(cls, item)
            field.process_record(data_record)
            features[field.name] = field.tensor

        for item in cls.DATA_FIELDS:
            if item in cls.FEATURES:
                continue

            field = getattr(cls, item)
            field.process_record(data_record)
            labels[field.name] = field.tensor

        return features, labels
