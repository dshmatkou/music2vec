import tensorflow as tf
from common.dataset_records import FeaturedRecord


def prepare_dataset(ds_path):
    dataset = tf.data.TFRecordDataset(
        ds_path
    ).map(
        FeaturedRecord.parse
    ).map(
        FeaturedRecord.split_features_labels
    ).shuffle(
        1000
    ).batch(
        100
    ).prefetch(
        100
    ).repeat()
    return dataset
