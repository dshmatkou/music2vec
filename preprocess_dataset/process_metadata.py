import logging
import os
import pandas

logger = logging.getLogger(__name__)


SET_SUBSET = ('set', 'subset')

TRACKS_COLUMNS = {
    ('set', 'subset'): 'subset',
    ('album', 'date_released'): 'release_date',
    ('set', 'split'): 'split',
    ('track', 'genres'): 'top_genres',
    ('track', 'genres_all'): 'genres_all',
}

ECHONEST_COLUMNS = {
    ('echonest', 'audio_features', 'acousticness'): 'acousticness',
    ('echonest', 'audio_features', 'danceability'): 'danceability',
    ('echonest', 'audio_features', 'energy'): 'energy',
    ('echonest', 'audio_features', 'instrumentalness'): 'instrumentalness',
    ('echonest', 'audio_features', 'speechiness'): 'speechiness',
    ('echonest', 'audio_features', 'valence'): 'happiness',
    ('echonest', 'metadata', 'artist_location'): 'artist_location',
}


def get_size_labels(size):
    if size == 'medium':
        return (size, 'small')
    elif size == 'large':
        return (size, 'medium', 'small')
    else:
        return (size,)


def get_metadata_fpath(data_dir, metadata_fn):
    return os.path.join(data_dir, 'fma_metadata', metadata_fn)


def extract_track_metadata(data_dir, dataset_size):
    track_metadata = pandas.read_csv(
        get_metadata_fpath(data_dir, 'tracks.csv'),
        index_col=0,
        header=[0, 1],
    )
    size_labels = get_size_labels(dataset_size)
    using_tracks_index = track_metadata[SET_SUBSET] == size_labels[0]
    for lbl in size_labels[1:]:
        using_tracks_index |= track_metadata[SET_SUBSET] == lbl
    using_tracks = track_metadata[using_tracks_index]
    using_tracks = using_tracks[list(TRACKS_COLUMNS.keys())]
    using_tracks.columns = using_tracks.columns.droplevel(0)
    return using_tracks


def extract_echonest_metadata(data_dir, using_tracks):
    echonest_metadata = pandas.read_csv(
        get_metadata_fpath(data_dir, 'raw_echonest.csv'),
        index_col=0,
        header=[0, 1, 2],
    )
    echonest_metadata = echonest_metadata[echonest_metadata.index.isin(using_tracks.index)]
    echonest_metadata = echonest_metadata[list(ECHONEST_COLUMNS.keys())]
    echonest_metadata.columns = echonest_metadata.columns.droplevel(0).droplevel(0)
    return echonest_metadata


def process_metadata(data_dir, dataset_size, enable_echonest=False):
    logger.info('Loading metadata from tracks.csv')
    tracks_metadata = extract_track_metadata(data_dir, dataset_size)
    if enable_echonest:
        logger.info('Loading metadata from raw_echonest.csv')
        echonest_metadata = extract_echonest_metadata(data_dir, tracks_metadata)
        tracks_metadata = tracks_metadata[tracks_metadata.index.isin(echonest_metadata.index)]
        logger.info('Merging rows')
        tracks_metadata = tracks_metadata.merge(
            echonest_metadata,
            left_index=True, right_index=True
        )
    return tracks_metadata
