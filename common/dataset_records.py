import tensorflow as tf
from common.records.fields import ScalarField, TensorField
from common.records.record import (
    BaseDataRecord
)


class BaseM2VRecord(BaseDataRecord):
    release_decade = TensorField(tf.float32, (11,))
    genres_top = TensorField(tf.float32, (161,))
    genres_all = TensorField(tf.float32, (161,))
    acousticness = ScalarField(tf.float32)
    danceability = ScalarField(tf.float32)
    energy = ScalarField(tf.float32)
    instrumentalness = ScalarField(tf.float32)
    speechiness = ScalarField(tf.float32)
    happiness = ScalarField(tf.float32)
    artist_location = TensorField(tf.float32, (230,))


class FeaturedRecord(BaseM2VRecord):
    FEATURES = {'feature'}
    feature = TensorField(tf.float32, (50, 1001, 1))


class VectorizedRecord(BaseM2VRecord):
    FEATURES = {'vector'}
    vector = TensorField(tf.float32, (1, 200))
