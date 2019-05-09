import logging
import os
import random
import shutil
import sys
import tensorflow as tf
from contextlib import ExitStack
from preprocess_dataset.audio.process import process_audio
from itertools import islice
from preprocess_dataset.metadata.process import process_metadata
from common.dataset_records import FeaturedRecord

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('__name__')


def get_full_output_name(output_dir, dataset_size, audio_processor):
    dataset_dir = os.path.join(output_dir, 'dataset-{}-{}'.format(
        dataset_size, audio_processor
    ))
    return dataset_dir


def batch_dataset(iterable, n=100):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


def write_dataset(dataset, output_path):
    with tf.python_io.TFRecordWriter(output_path) as tfwriter:
        iterator = dataset.make_one_shot_iterator()
        for record in iterator:
            tfwriter.write(record)


def main(
    dataset_dir,
    dataset_size,
    audio_processor,
    output_dir,
    test_size,
    validate_size
):
    logger.info('Start processing')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info('Start processing metadata')
    tracks_metadata = process_metadata(
        dataset_dir,
        dataset_size,
    )

    output_name = get_full_output_name(
        output_dir,
        dataset_size,
        audio_processor,
    )

    if os.path.exists(output_name):
        shutil.rmtree(output_name)
    os.mkdir(output_name)

    train_fn = os.path.join(output_name, 'train.tfrecord')
    test_fn = os.path.join(output_name, 'test.tfrecord')
    validate_fn = os.path.join(output_name, 'validate.tfrecord')

    logger.info('Start batch processing')

    with ExitStack() as stack:
        train_writer = stack.enter_context(tf.python_io.TFRecordWriter(train_fn))
        test_writer = stack.enter_context(tf.python_io.TFRecordWriter(test_fn))
        validate_writer = stack.enter_context(tf.python_io.TFRecordWriter(validate_fn))

        for bn, batch in enumerate(batch_dataset(tracks_metadata.items())):
            logger.info('Processing %s batch', bn)
            batch = dict(batch)

            logger.info('Start processing audio')
            batch = process_audio(
                dataset_dir,
                batch,
                audio_processor,
            )

            logger.info('Serializing batch')
            serialized = [
                FeaturedRecord.serialize(record)
                for record in batch.values()
            ]

            logger.info('Writing data')
            for item in serialized:
                rand = random.random()
                if rand <= validate_size:
                    validate_writer.write(item)
                elif validate_size < rand <= (validate_size + test_size):
                    test_writer.write(item)
                else:
                    train_writer.write(item)

            train_writer.flush()
            test_writer.flush()
            validate_writer.flush()

    logger.info('Finished')
