import logging
import os
import pandas
from audio_processors import get_processor

logger = logging.getLogger(__name__)


def get_audio_path(dataset_dir, dataset_size, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(dataset_dir, 'fma_' + dataset_size, tid_str[:3], tid_str + '.mp3')


def process_audio(dataset_dir, audio_metadata, proc_name):
    logger.info('Getting audio processor %s', proc_name)
    processor = get_processor(proc_name)
    new_df_raw = {'track_id': [], 'features': []}
    for track_id, row in audio_metadata.iterrows():
        logger.info('Processing audio %s', track_id)
        audio_path = get_audio_path(dataset_dir, row['subset'], track_id)
        features = processor(audio_path).tolist()
        new_df_raw['features'].append(features)
        new_df_raw['track_id'].append(track_id)

    audio_data = pandas.DataFrame(new_df_raw)
    audio_data = audio_data.set_index('track_id')

    result_dataset = audio_metadata.merge(
        audio_data,
        left_index=True, right_index=True
    )

    return result_dataset
