import logging
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
    ).shuffle()
    train = dataset.take(500)
    test = dataset.skip(500)

    return train, test


def parse_args(parser):
    parser.add_argument('--dataset', help='Path to dataset')
    args = parser.parse_args()
    return args


def main(dataset):

    logger.info('Start')
    train, test = prepare_dataset(dataset)

    logger.info('Train')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=20,
            model_dir='/tmp/music2vec_models',
        )
    )
    estimator.train(train.batch(100).shuffle(reshuffle_each_iteration=True), steps=5)

    logger.info('Test')
    e = estimator.evaluate(train.batch(100))
    print("Testing Accuracy:", e['accuracy'])
