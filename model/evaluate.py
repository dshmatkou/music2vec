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
    vectors = estimator.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x=features,
            shuffle=False
        )
    )
    for item, vector in zip(dataset, vectors):
        item.pop('feature')
        item['vector'] = vector
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
    top_genres_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    logger.info('Fit knn')
    top_genres_knn.fit(
        [item['vector'] for item in test_dataset],
        [extract_multilabel(item['genres_top']) for item in test_dataset]
    )
    logger.info('Evaluate')
    score = top_genres_knn.score(
        [item['vector'] for item in eval_dataset],
        [extract_multilabel(item['genres_top']) for item in eval_dataset]
    )
    print('Score: {}', score)
