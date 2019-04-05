import json
import logging
import pandas
import six
from dateutil.parser import parse
from common.records.record import Record
from preprocess_dataset.metadata.processors.base import CategoricalColumnProcessor

logger = logging.getLogger(__name__)


SET_SUBSET = ('set', 'subset')


TRACKS_COLUMNS = {
    ('album', 'date_released'): 'release_date',
    ('track', 'genres'): 'top_genres',
    ('track', 'genres_all'): 'genres_all',
}


class ReleaseDateColumn(CategoricalColumnProcessor):
    NAME = 'release_decade'
    DF_COLUMN = ('album', 'date_released')

    def _process_raw_column(self, raw_value):
        if isinstance(raw_value, six.string_types):
            decade = parse(raw_value).year // 10
            return [decade], str(decade) + '0s'
        return [None], 'unknown'


class CommonGenresColumn(CategoricalColumnProcessor):

    def _process_raw_column(self, raw_value):
        value = json.loads(raw_value)
        return value, None


class TopGenresColumn(CommonGenresColumn):
    NAME = 'genres_top'
    DF_COLUMN = ('track', 'genres')


class AllGenresColumn(CommonGenresColumn):
    NAME = 'genres_all'
    DF_COLUMN = ('track', 'genres_all')


COLUMNS = [
    ReleaseDateColumn,
    TopGenresColumn,
    AllGenresColumn,
]


def get_size_labels(size):
    if size == 'medium':
        return (size, 'small')
    elif size == 'large':
        return (size, 'medium', 'small')
    else:
        return (size,)


def extract_track_metadata(dataset_fname, dataset_size):
    dataset = {}

    track_metadata = pandas.read_csv(dataset_fname, index_col=0, header=[0, 1])

    logger.info('Create column processors')
    COLUMNS_PROCESSORS = [
        column_cls(track_metadata)
        for column_cls in COLUMNS
    ]

    size_labels = get_size_labels(dataset_size)

    logger.info('Filter records')
    using_tracks_index = track_metadata[SET_SUBSET] == size_labels[0]
    for lbl in size_labels[1:]:
        using_tracks_index |= track_metadata[SET_SUBSET] == lbl
    using_tracks = track_metadata[using_tracks_index]

    logger.info('Processing records')
    for index, row in using_tracks.iterrows():
        dataset[index] = record = Record()
        record.track_id = index
        record.subset = row[SET_SUBSET]

        for processor in COLUMNS_PROCESSORS:
            dataset[index].update(processor.process_item(index))

        dataset[index]['subset'] = row[SET_SUBSET]

    return dataset
