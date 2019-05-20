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
        ).prefetch(
            20
        ).batch(
            20
        ).shuffle(
            10
        )
        return dataset
