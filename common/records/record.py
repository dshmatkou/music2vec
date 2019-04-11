from common.records.base import (
    BaseDataRecord,
    FloatScalarColumn,
    Tensor1DColumn,
    Tensor2DColumn,
)


class Record(BaseDataRecord):
    FEATURES = {'feature'}

    release_decade = Tensor1DColumn()
    genres_top = Tensor1DColumn()
    genres_all = Tensor1DColumn()
    acousticness = FloatScalarColumn()
    danceability = FloatScalarColumn()
    energy = FloatScalarColumn()
    instrumentalness = FloatScalarColumn()
    speechiness = FloatScalarColumn()
    happiness = FloatScalarColumn()
    artist_location = Tensor1DColumn()
    feature = Tensor2DColumn()

    def __init__(self):
        super().__init__()
        self.track_id = None
        self.subset = None

    def update(self, new_items):
        """
        :type new_items: dict
        """
        for key, value in new_items.items():
            if key in self.DATA_FIELDS:
                self[key] = value
