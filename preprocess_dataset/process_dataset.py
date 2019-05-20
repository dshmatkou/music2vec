import logging
import os
import random
import shutil
import sys
import tensorflow as tf
from contextlib import ExitStack
from preprocess_dataset.audio.process import process_audio
from preprocess_dataset.metadata.process import process_metadata
from common.dataset_records import FeaturedRecord
from common.records.exceptions import RecordInvalidShapesError

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('__name__')


def get_full_output_name(output_dir, dataset_size, audio_processor):
    dataset_dir = os.path.join(output_dir, 'dataset-{}-{}'.format(
        dataset_size, audio_processor
    ))
    return dataset_dir


def batch_dataset(iterable, n=20):
    batches = []
    batch = []
    for item in iterable:
        if len(batch) == n:
            batches.append(batch)
            batch = []
        batch.append(item)

    if batch:
        batches.append(batch)

    return batches


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

        total_count = len(tracks_metadata)
        batched_dataset = batch_dataset(tracks_metadata.items())
        del tracks_metadata

        processed = 0
        for bn in range(len(batched_dataset)):
            logger.info('Processing %s batch', bn)
            batch = batched_dataset[bn]
            batched_dataset[bn] = None
            batch = dict(batch)

            logger.info('Start processing audio')
            batch = process_audio(
                dataset_dir,
                batch,
                audio_processor,
            )

            logger.info('Serializing batch')

            serialized = []
            for track_id, track_features in batch.items():
                if 'feature' not in track_features:
                    logger.warning('Record has not been featured %s', track_id)
                    continue

                try:
                    record = FeaturedRecord.serialize(track_features)
                    serialized.append(record)
                except RecordInvalidShapesError as ex:
                    logger.warning('Record has wrong data: %s', str(ex))

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

            processed += len(batch)
            logger.info('Processed %s / %s', processed, total_count)
            del batch
            del serialized

    logger.info('Finished')
