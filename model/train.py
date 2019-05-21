import logging
import os
import sys
import tensorflow as tf
from common.dataset_records import FeaturedRecord
from model.model import model_fn
from model.utils import prepare_dataset

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

EPOCHS = 10


def input_serving_fn():
    input_shape = (None,) + FeaturedRecord.feature.shape
    inputs = {'feature': tf.placeholder(tf.float32, input_shape)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main(dataset, output_dir):
    logger.info('Start')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=50,
            model_dir='/tmp/music2vec_models',
        )
    )

    logger.info('Load dataset')
    train_path = os.path.join(dataset, 'train.tfrecord')
    test_path = os.path.join(dataset, 'test.tfrecord')

    for epoch in range(EPOCHS):
        logger.info('Epoch %s/%s', epoch + 1, EPOCHS)
        logger.info('Train')
        estimator.train(
            input_fn=lambda: prepare_dataset(train_path),
            steps=150,
        )
        logger.info('Test')
        e = estimator.evaluate(lambda: prepare_dataset(test_path))
        logger.info('Testing Accuracy: %s', e)
        logger.info('Save model')
        estimator.export_savedmodel(
            export_dir_base=output_dir,
            serving_input_receiver_fn=input_serving_fn,
        )
        logger.info('Saved')
    logger.info('Finish')
