import argparse
import logging
import sys
import tensorflow as tf
from model import model_fn

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def extract_fn(data_record):
    features = {
        'release_decade': tf.FixedLenFeature([2], tf.int64),
        'release_decade_raw': tf.FixedLenFeature([], tf.string, default_value=''),
        'genres_top': tf.FixedLenFeature([], tf.int64),
        'genres_all': tf.FixedLenFeature([], tf.int64),
        'acousticness': tf.FixedLenFeature([1], tf.float32),
        'danceability': tf.FixedLenFeature([1], tf.float32),
        'energy': tf.FixedLenFeature([1], tf.float32),
        'instrumentalness': tf.FixedLenFeature([1], tf.float32),
        'speechiness': tf.FixedLenFeature([1], tf.float32),
        'happiness': tf.FixedLenFeature([1], tf.float32),
        'artist_location': tf.FixedLenFeature([2], tf.int64),
        'artist_location_raw': tf.FixedLenFeature([1], tf.string, default_value=''),
        'feature_shape': tf.FixedLenFeature([2], tf.int64),
        'feature': tf.VarLenFeature(tf.float32),
    }
    sample = tf.parse_single_example(data_record, features)
    import ipdb; ipdb.set_trace()
    sample['feature'] = tf.reshape(
        sample['feature'],
        (
            tf.cast(sample['feature_shape'][0], tf.int64),
            tf.cast(sample['feature_shape'][1], tf.int64),
        ),
    )
    return (
        {'feature': sample['feature']},
        {
            'release_decade': sample['release_decade'],
            'genres_top': sample['genres_top'],
            'genres_all': sample['genres_all'],
            'acousticness': sample['acousticness'],
            'danceability': sample['danceability'],
            'energy': sample['energy'],
            'instrumentalness': sample['instrumentalness'],
            'speechiness': sample['speechiness'],
            'happiness': sample['happiness'],
            'artist_location': sample['artist_location'],
        },
    )


def prepare_dataset(ds_path):
    dataset = tf.data.TFRecordDataset([ds_path])
    dataset = dataset.map(extract_fn).shuffle()
    train = dataset.take(500)
    test = dataset.skip(500)

    return train, test


def parse_args(parser):
    parser.add_argument('--dataset', help='Path to dataset')
    args = parser.parse_args()
    return args


def main(parser):
    args = parse_args(parser)

    logger.info('Start')
    train, test = prepare_dataset(args.dataset)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
