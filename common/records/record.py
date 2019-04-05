import numpy as np
import tensorflow as tf
from common.records.base import BaseDataRecord, RecordColumn


class FloatColumn(RecordColumn):
    TYPE = 'float'

    def process_raw_value(self, value):
        return float(value)

    @property
    def descriptor(self):
        return tf.FixedLenFeature([], tf.float32)

    def parse(self, item):
        return item


class MultiLabelColumn(RecordColumn):
    """
    >>> class Record(BaseDataRecord):
    >>>     lbl = MultiLabelColumn('lbl')
    >>>
    >>> labels_count, target_labels = 10, [1]
    >>> rec = Record()
    >>> rec.lbl = labels_count, target_labels
    """

    def process_raw_value(self, value):
        labels_count, labels = value
        serialized = np.array([labels_count] + labels).tobytes()
        return serialized

    def parse(self, item):
        import ipdb; ipdb.set_trace()
        tensor = np.frombuffer(item.numpy())
        labels_count = tensor[0]
        labels = {tensor[i] for i in range(1, tensor.shape[0])}
        vector = [
            (1 if i in labels else 0)
            for i in range(labels_count)
        ]
        return vector


class TensorColumn(RecordColumn):
    def process_raw_value(self, value):
        serialized = np.array(value).tobytes()
        return serialized

    def parse(self, item):
        import ipdb; ipdb.set_trace()
        return np.frombuffer(item.numpy())


class Record(BaseDataRecord):
    FEATURES = {'feature'}

    release_decade = MultiLabelColumn()
    genres_top = MultiLabelColumn()
    genres_all = MultiLabelColumn()
    acousticness = FloatColumn()
    danceability = FloatColumn()
    energy = FloatColumn()
    instrumentalness = FloatColumn()
    speechiness = FloatColumn()
    happiness = FloatColumn()
    artist_location = MultiLabelColumn()
    feature = TensorColumn()

    def __init__(self):
        super().__init__()
        self.track_id = None
        self.subset = None

    def update(self, new_items):
        """
        :type new_items: dict
        """
        for key, value in new_items.items():
            if key in self.DATA_FIELDS:
                self[key] = value
