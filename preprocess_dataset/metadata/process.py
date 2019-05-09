import logging
import os
from preprocess_dataset.metadata.processors.tracks import extract_track_metadata
from preprocess_dataset.metadata.processors.echonest import extract_echonest_metadata

logger = logging.getLogger(__name__)


def get_metadata_fpath(data_dir, metadata_fn):
    return os.path.join(data_dir, 'fma_metadata', metadata_fn)


def process_metadata(data_dir, dataset_size, enable_echonest=True):
    logger.info('Loading metadata from tracks.csv')
    tracks_metadata = extract_track_metadata(
        get_metadata_fpath(data_dir, 'tracks.csv'),
        dataset_size,
    )
    if enable_echonest:
        logger.info('Loading metadata from raw_echonest.csv')
        tracks_metadata = extract_echonest_metadata(
            get_metadata_fpath(data_dir, 'raw_echonest.csv'),
            tracks_metadata
        )

    return tracks_metadata
