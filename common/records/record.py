import tensorflow as tf
from common.records.exceptions import FieldInvalidShapeError, RecordInvalidShapesError
from common.records.fields import BaseField


class RecordMeta(type):
    DF_ATTR_NAME = 'DATA_FIELDS'

    def __new__(mcs, name, bases, attrs):
        if mcs.DF_ATTR_NAME not in attrs:
            attrs[mcs.DF_ATTR_NAME] = set()

        for attr_name, attr in attrs.items():
            if not isinstance(attr, BaseField):
                continue
            attrs[mcs.DF_ATTR_NAME].add(attr_name)
            attr.name = attr_name

        for base in bases:
            if not isinstance(base, RecordMeta):
                continue
            attrs[mcs.DF_ATTR_NAME] |= getattr(base, mcs.DF_ATTR_NAME)

        return super().__new__(mcs, name, bases, attrs)


class BaseDataRecord(object, metaclass=RecordMeta):
    DATA_FIELDS = set()
    FEATURES = set()

    @classmethod
    def serialize(self, data):
        exceptions = []
        features = {}
        for field_name in self.DATA_FIELDS:
            try:
                field = getattr(self, field_name)
                feature = field.process_raw(data[field_name])
                features.update(feature)
            except FieldInvalidShapeError as ex:
                exceptions.append(ex)

        if exceptions:
            msg = '\n'.join([str(ex) for ex in exceptions])
            raise RecordInvalidShapesError(msg)

        record = tf.train.Features(feature=features)
        example = tf.train.Example(features=record)
        return example.SerializeToString()

    @classmethod
    def get_descriptor(cls):
        record_description = {}
        for item in cls.DATA_FIELDS:
            field = getattr(cls, item)
            record_description.update(field.descriptor)

        return record_description

    @classmethod
    def parse(cls, example_proto):
        example = tf.parse_single_example(example_proto, cls.get_descriptor())
        return example

    @classmethod
    def split_features_labels(cls, data_dict):
        features, labels = {}, {}
        for item in cls.FEATURES:
            features[item] = data_dict[item]

        for item in cls.DATA_FIELDS:
            if item in cls.FEATURES:
                continue

            labels[item] = data_dict[item]

        return features, labels

    @classmethod
    def unpack_str(cls, buffer):
        example = tf.train.Example()
        example.ParseFromString(buffer)
        record = {}
        for field_name in cls.DATA_FIELDS:
            field = getattr(cls, field_name)
            record[field_name] = field.unpack(example.features.feature[field_name])

        return record
