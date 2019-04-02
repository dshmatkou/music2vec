import argparse
import logging
import os
import sys
import tensorflow as tf
from audio.process import process_audio
from audio.processors import PROCESSORS
from itertools import islice
from metadata.process import process_metadata

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


def parse_args(parser):
    parser.add_argument('--dataset-dir', help='Directory with dataset')
    parser.add_argument(
        '--dataset-size',
        choices=['large', 'medium', 'small'],
        help='Size of processing dataset',
    )
    parser.add_argument(
        '--audio-processor',
        choices=list(PROCESSORS.keys()),
        help='Audio preprocessor',
    )
    parser.add_argument('--output-dir', help='Output directory', default='')
    parser.add_argument(
        '--with-echonest',
        action='store_true',
        default=False,
        help='Enable echonest features (may reduce result dataset)',
    )
    args = parser.parse_args()
    return args


def main(parser):
    logger.info('Start processing')
    args = parse_args(parser)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger.info('Start processing metadata')
    tracks_metadata = process_metadata(
        args.dataset_dir,
        args.dataset_size,
        args.with_echonest,
    )

    output_name = get_full_output_name(
        args.output_dir,
        args.dataset_size,
        args.audio_processor,
        args.with_echonest,
    ) + '.tfrecord'

    logger.info('Start batch processing')
    with tf.python_io.TFRecordWriter(output_name) as tfwriter:
        for bn, batch in enumerate(batch_dataset(tracks_metadata.items())):
            logger.info('Processing %s batch', bn)
            batch = dict(batch)

            logger.info('Start processing audio')
            batch = process_audio(
                args.dataset_dir,
                batch,
                args.audio_processor,
            )

            logger.info('Start writing data')
            for item_id, features in batch.items():
                features.pop('subset', None)
                record = tf.train.Features(feature=features)
                example = tf.train.Example(features=record)
                tfwriter.write(example.SerializeToString())

            tfwriter.flush()
            break

    logger.info('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
