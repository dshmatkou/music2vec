import argparse
import collections
import json
import logging
import numpy
import pandas
import six
import sys
import tensorflow as tf
from dateutil.parser import parse
from model import model_fn
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def genres_extractor(row, col_name='genres_all'):
    return set(json.loads(row[col_name]))


def location_extractor(row):
    return {str(row['artist_location']).split(',')[-1].strip()}


def create_mapping(df, extractor):
    all_genres = set()
    for idx, row in df.iterrows():
        all_genres |= extractor(row)

    mapping = {}
    for i, g in enumerate(all_genres):
        mapping[g] = i

    return mapping


def to_categorical(mapping, values):
    nv = {mapping[value] for value in values}
    cat = [1 if category in nv else 0 for category in range(len(mapping))]
    return cat


def process_rows(df, genres_mapping, locations_mapping):
    logger.info('Process features labels')
    features = collections.defaultdict(list)
    labels = collections.defaultdict(list)

    for idx, row in df.iterrows():
        feature = numpy.array(json.loads(row['features']))
        features['feature'].append(feature.reshape(feature.shape + (1,)))
        labels['date_released'].append(
            [parse(row['date_released']).year]
            if isinstance(row['date_released'], six.string_types)
            else [2000]
        )
        labels['genres_all'].append(to_categorical(genres_mapping, genres_extractor(row)))
        labels['genres'].append(
            to_categorical(genres_mapping, genres_extractor(row, 'genres'))
        )

        if 'acousticness' in row:
            labels['acousticness'].append([row['acousticness']])
            labels['danceability'].append([row['danceability']])
            labels['energy'].append([row['energy']])
            labels['instrumentalness'].append([row['instrumentalness']])
            labels['speechiness'].append([row['speechiness']])
            labels['valence'].append([row['valence']])
            labels['artist_location'].append(
                to_categorical(locations_mapping, location_extractor(row))
            )

    for key, value in features.items():
        features[key] = numpy.array(value, dtype=numpy.float32)

    for key, value in labels.items():
        labels[key] = numpy.array(value, dtype=numpy.float32)

    return features, labels


def prepare_dataset(ds_path):
    logger.info('Preparing dataset')
    df = pandas.read_csv(ds_path)
    logger.info('Create mappings')
    genres_mapping = create_mapping(df, genres_extractor)
    locations_mapping = create_mapping(df, location_extractor)
    logger.info('Processing')
    train, test = train_test_split(df, test_size=0.2)
    x_train, y_train = process_rows(train, genres_mapping, locations_mapping)
    x_test, y_test = process_rows(test, genres_mapping, locations_mapping)

    return x_train, y_train, x_test, y_test


def parse_args(parser):
    parser.add_argument('--dataset', help='Path to dataset')
    args = parser.parse_args()
    return args


def main(parser):
    args = parse_args(parser)

    logger.info('Start')
    x_train, y_train, x_test, y_test = prepare_dataset(args.dataset)

    logger.info('Train')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=20,
            model_dir='/tmp/music2vec_models',
        )
    )
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_train, y=y_train,
        batch_size=100, num_epochs=None, shuffle=True
    )
    estimator.train(input_fn, steps=5)

    logger.info('Test')
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_test, y=y_test,
        batch_size=100, shuffle=False
    )
    e = estimator.evaluate(input_fn)
    print("Testing Accuracy:", e['accuracy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
