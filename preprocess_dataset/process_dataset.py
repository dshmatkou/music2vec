import logging
import os
import shutil
import sys
import tempfile
import tensorflow as tf
from preprocess_dataset.audio.process import process_audio
from itertools import islice
from preprocess_dataset.metadata.process import process_metadata

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
        for record in dataset.make_one_shot_iterator():
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

    logger.info('Start batch processing')
    with tempfile.NamedTemporaryFile(suffix='.tfrecord') as tmpf:
        with tf.python_io.TFRecordWriter(tmpf.name) as tfwriter:
            for bn, batch in enumerate(batch_dataset(tracks_metadata.items())):
                logger.info('Processing %s batch', bn)
                batch = dict(batch)

                logger.info('Start processing audio')
                batch = process_audio(
                    dataset_dir,
                    batch,
                    audio_processor,
                )

                logger.info('Start writing data')
                for item_id, features in batch.items():
                    record = tf.train.Features(feature=features)
                    example = tf.train.Example(features=record)
                    tfwriter.write(example.SerializeToString())

                tfwriter.flush()
                break

        logger.info('Split train test')
        dataset = tf.data.TFRecordDataset([tmpf.name])
        train_size = len(tracks_metadata) * (1 - test_size)

        dataset.shuffle(1000, 1234)
        train = dataset.take(train_size)
        test = dataset.skip(train_size)

        if os.path.exists(output_name):
            shutil.rmtree(output_name)

        os.mkdir(output_name)
        write_dataset(train, os.path.join(output_name, 'train.tfrecord'))
        write_dataset(test, os.path.join(output_name, 'test.tfrecord'))

    logger.info('Finished')
