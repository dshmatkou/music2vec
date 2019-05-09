import logging
import os
import sys
import tensorflow as tf
from model.model import model_fn
from model.utils import prepare_dataset

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

EPOCHS = 10


def main(dataset):
    logger.info('Start')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=100,
            model_dir='/tmp/music2vec_models',
        )
    )

    logger.info('Load dataset')
    train_path = os.path.join(dataset, 'train.tfrecord')
    test_path = os.path.join(dataset, 'test.tfrecord')

    train_dataset = prepare_dataset(train_path)
    test_dataset = prepare_dataset(test_path)

    for epoch in range(EPOCHS):
        logger.info('Epoch %s/%s', epoch + 1, EPOCHS)
        logger.info('Train')
        estimator.train(
            input_fn=lambda: train_dataset.make_one_shot_iterator(),
        )
        logger.info('Test')
        e = estimator.evaluate(lambda: test_dataset.make_one_shot_iterator())
        logger.info("Testing Accuracy: %s", e)

    logger.info('Finish')
