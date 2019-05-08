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


def get_full_output_name(output_dir, dataset_size, audio_processor, with_echonest):
    dataset_dir = os.path.join(output_dir, 'dataset-{}-{}-{}-echonest'.format(
        dataset_size, audio_processor, 'with' if with_echonest else 'no'
    ))
    return dataset_dir


def batch_dataset(iterable, n=10):
    i = iter(iterable)
    piece = islice(i, n)
    while piece:
        yield piece
        piece = islice(i, n)


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
    with_echonest,
    test_size
):
    logger.info('Start processing')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info('Start processing metadata')
    tracks_metadata = process_metadata(
        dataset_dir,
        dataset_size,
        with_echonest,
    )

    output_name = get_full_output_name(
        output_dir,
        dataset_size,
        audio_processor,
        with_echonest,
    )

    if os.path.exists(output_name):
        shutil.rmtree(output_name)
    os.mkdir(output_name)

    train_fn = os.path.join(output_name, 'train.tfrecord')
    test_fn = os.path.join(output_name, 'test.tfrecord')

    logger.info('Start batch processing')

    with ExitStack() as stack:
        train_writer = stack.enter_context(tf.python_io.TFRecordWriter(train_fn))
        test_writer = stack.enter_context(tf.python_io.TFRecordWriter(test_fn))
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
                if random.random() > test_size:
                    train_writer.write(item)
                else:
                    test_writer.write(item)

            train_writer.flush()
            test_writer.flush()
            break

    logger.info('Finished')
