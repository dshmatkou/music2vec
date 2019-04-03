import logging
import pandas
import tensorflow as tf
from preprocess_dataset.metadata.processors.base import CategoricalColumnProcessor

logger = logging.getLogger(__name__)


ECHONEST_COLUMNS = {
    ('echonest', 'audio_features', 'acousticness'): 'acousticness',
    ('echonest', 'audio_features', 'danceability'): 'danceability',
    ('echonest', 'audio_features', 'energy'): 'energy',
    ('echonest', 'audio_features', 'instrumentalness'): 'instrumentalness',
    ('echonest', 'audio_features', 'speechiness'): 'speechiness',
    ('echonest', 'audio_features', 'valence'): 'happiness',
}


class ArtistLocationColumn(CategoricalColumnProcessor):
    NAME = 'artist_location'
    DF_COLUMN = ('echonest', 'metadata', 'artist_location')

    def _process_raw_column(self, raw_value):
        result = str(raw_value).split(',')[-1].strip()
        return [result], result


def extract_echonest_metadata(dataset_fname, using_tracks):
    """
    :param dataset_fname: str
    :param using_tracks: dict
    :return:
    """
    dataset = {}
    logger.info('Loading echonest metadata from file')
    echonest_metadata = pandas.read_csv(dataset_fname, index_col=0, header=[0, 1, 2])
    artist_location_column_processor = ArtistLocationColumn(echonest_metadata)

    logger.info('Filter using records')
    echonest_metadata = echonest_metadata[echonest_metadata.index.isin(using_tracks.keys())]

    logger.info('Processing records')
    for index, row in echonest_metadata.iterrows():
        dataset[index] = using_tracks[index]

        for df_column, column in ECHONEST_COLUMNS.items():
            col_value = row[df_column]
            dataset[index][column] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(col_value)])
            )

        dataset[index].update(
            artist_location_column_processor.process_item(index)
        )

    return dataset
