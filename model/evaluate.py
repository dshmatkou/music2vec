import logging
import numpy as np
import os
import sklearn
import sys
import tensorflow as tf
from common.dataset_records import FeaturedRecord
from model.model import model_fn

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def vectorize_dataset(dataset_fn, estimator):
    dataset = [
        FeaturedRecord.unpack_str(buffer)
        for buffer in tf.python_io.tf_record_iterator(dataset_fn)
    ]
    features = {'feature': np.array([item['feature'] for item in dataset])}
    predictions = estimator.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x=features,
            shuffle=False
        )
    )
    for item, prediction in zip(dataset, predictions):
        for key in item:
            if key not in prediction:
                item.pop(key)
        item['vector'] = prediction['vector']
    return dataset


def extract_multilabel(label):
    labels = [i for i, item in enumerate(label.tolist()) if round(item) == 1]
    return labels


METRICS = {
    'genres_all': sklearn.linear_model.LogisticRegression(multi_class='multinomial'),
    'genres_top': sklearn.linear_model.LogisticRegression(multi_class='multinomial'),
    'release_decade': sklearn.linear_model.LogisticRegression(),
    'acousticness': sklearn.linear_model.LinearRegression(),
    'danceability': sklearn.linear_model.LinearRegression(),
    'energy': sklearn.linear_model.LinearRegression(),
    'instrumentalness': sklearn.linear_model.LinearRegression(),
    'speechiness': sklearn.linear_model.LinearRegression(),
    'happiness': sklearn.linear_model.LinearRegression(),
    'artist_location': sklearn.linear_model.LogisticRegression(),
}


def evaluate(dataset):
    logger.info('Load dataset')
    test_path = os.path.join(dataset, 'test.tfrecord')
    eval_path = os.path.join(dataset, 'validate.tfrecord')

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=10,
            model_dir='/tmp/music2vec_models',
        )
    )

    test_dataset = vectorize_dataset(test_path, estimator)
    eval_dataset = vectorize_dataset(eval_path, estimator)

    for name, metric_learner in METRICS.items():
        if name not in test_dataset[0]:
            continue

        logger.info('Check: %s', name)
        metric_learner.fit(
            np.array([item['vector'] for item in test_dataset]),
            np.array([item[name] for item in test_dataset], dtype=np.int64),
        )
        logger.info('Evaluate')
        score = metric_learner.score(
            np.array([item['vector'] for item in eval_dataset]),
            np.array([item[name] for item in eval_dataset], dtype=np.int64),
        )
        print('Score:', score)
