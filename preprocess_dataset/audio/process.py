import logging
import os
import tensorflow as tf
from preprocess_dataset.audio.processors import get_processor

logger = logging.getLogger(__name__)


def get_audio_path(dataset_dir, dataset_size, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(dataset_dir, 'fma_' + dataset_size, tid_str[:3], tid_str + '.mp3')


def process_audio(dataset_dir, audio_metadata, proc_name):
    """
    :param dataset_dir: str
    :param audio_metadata: dict[int, dict]
    :param proc_name: str
    :return: dict[int, dict]
    """
    logger.info('Getting audio processor %s', proc_name)
    processor = get_processor(proc_name)

    for index, track_id in enumerate(audio_metadata.keys()):
        logger.info(
            'Processing audio %s, %s/%s',
            track_id, index,
            len(audio_metadata),
        )
        audio_path = get_audio_path(
            dataset_dir, audio_metadata[track_id]['subset'], track_id,
        )
        features = processor(audio_path)

        rows, cols = features.shape
        audio_metadata[track_id]['feature_shape'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=features.shape)
        )
        audio_metadata[track_id]['feature'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features.reshape((rows * cols,)))
        )

    return audio_metadata
