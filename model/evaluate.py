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
        item.pop('feature')
        item['vector'] = prediction['vector']
    return dataset


def extract_multilabel(label):
    labels = [i for i, item in enumerate(label.tolist()) if round(item) == 1]
    return labels


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

    logger.info('Check top genres')
    top_genres_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
    logger.info('Fit knn')
    top_genres_knn.fit(
        np.array([item['vector'] for item in test_dataset]),
        np.array([item['genres_top'] for item in test_dataset], dtype=np.int64),
    )
    logger.info('Evaluate')
    score = top_genres_knn.score(
        np.array([item['vector'] for item in eval_dataset]),
        np.array([item['genres_top'] for item in eval_dataset], dtype=np.int64),
    )
    print('Score:', score)
