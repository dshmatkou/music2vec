import os
from preprocess_dataset.audio_processors import get_processor


def get_audio_path(dataset_dir, dataset_size, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(dataset_dir, 'fma_' + dataset_size, tid_str + '.mp3')


def process_audio(dataset_dir, audio_metadata, proc_name):
    processor = get_processor(proc_name)
    for track_id, row in audio_metadata.iterrows():
        audio_path = get_audio_path(dataset_dir, row['subset'], track_id)
        features = processor(audio_path)
        row['feature'] = features
        audio_metadata[track_id] = row

    return audio_metadata
