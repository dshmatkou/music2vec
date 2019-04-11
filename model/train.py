import logging
import os
import sys
import tensorflow as tf
from common.records.record import Record
from model.model import model_fn

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def prepare_dataset(ds_path):
    dataset = tf.data.TFRecordDataset([ds_path])
    dataset = dataset.map(
        Record.deserialize
    ).map(
        Record.decompose_on_feature_labels
    ).shuffle(1000, 54321, reshuffle_each_iteration=True)
    return dataset


def parse_args(parser):
    parser.add_argument('--dataset', help='Path to dataset')
    args = parser.parse_args()
    return args


def main(dataset):
    logger.info('Start')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=20,
            model_dir='/tmp/music2vec_models',
        )
    )

    logger.info('Load dataset')
    train_path = os.path.join(dataset, 'train.tfrecord')
    test_path = os.path.join(dataset, 'test.tfrecord')

    logger.info('Train')
    estimator.train(
        input_fn=lambda: prepare_dataset(train_path),
        steps=5
    )

    logger.info('Test')
    e = estimator.evaluate(lambda: prepare_dataset(test_path))
    print("Testing Accuracy:", e['accuracy'])
