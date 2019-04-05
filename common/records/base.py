import tensorflow as tf


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

    TYPE = None
    TRANSFORMERS = {
        'string': _bytes_feature,
        'int': _int64_feature,
        'float': _float_feature,
    }

    def __init__(self):
        self.name = None
        self.value = None

    def process_raw_value(self, value):
        return bytes(value)

    def __set__(self, instance, value):
        serializer = self.TRANSFORMERS.get(self.TYPE, _bytes_feature)
        self.value = serializer(self.process_raw_value(value))

    @property
    def descriptor(self):
        return tf.FixedLenFeature([], tf.string, default_value='')

    def parse(self, item):
        return item.numpy()


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
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def _collect_data(self):
        data = {}
        for item in self.DATA_FIELDS:
            field = self[item]
            data[field.name] = field.value

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
            record_description[field.name] = field.descriptor

        sample = tf.parse_single_example(data_record, record_description)
        return sample

    @classmethod
    def decompose_on_feature_labels(cls, data_record):
        features, labels = {}, {}
        for item in cls.FEATURES:
            field = getattr(cls, item)
            features[field.name] = field.parse(data_record[field.name])

        for item in cls.DATA_FIELDS:
            if item in cls.FEATURES:
                continue

            field = getattr(cls, item)
            labels[field.name] = field.parse(data_record[field.name])

        return features, labels
