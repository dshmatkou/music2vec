import tensorflow as tf
from common.dataset_records import FeaturedRecord


def prepare_dataset(ds_path):
    with tf.variable_scope('dataset'):
        dataset = tf.data.TFRecordDataset(
            ds_path
        ).map(
            FeaturedRecord.parse
        ).map(
            FeaturedRecord.split_features_labels
        ).shuffle(
            1000
        ).prefetch(
            10
        ).batch(
            10
        )
        return dataset
