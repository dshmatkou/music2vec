import tensorflow as tf


def to_categorical(mapping, values):
    nv = {mapping[value] for value in values}
    return len(mapping), list(nv)


class CategoricalColumnProcessor(object):
    NAME = None
    DF_COLUMN = None

    def __init__(self, df):
        """
        :param df: pandas.DataFrame
        """
        self.df = df

        self._all_values = set()

        for index, row in df.iterrows():
            raw_column = row[self.DF_COLUMN]
            self._register_raw_column_value(raw_column)

        self._mapping = {
            item: index
            for index, item in
            enumerate(self._all_values)
        }

    def _process_raw_column(self, raw_value):
        return [raw_value], None

    def _register_raw_column_value(self, raw_value):
        value, _ = self._process_raw_column(raw_value)
        for v in value:
            self._all_values.add(v)

    def process_item(self, item_id):
        result = {}

        values, hr_value = self._process_raw_column(
            self.df.ix[item_id][self.DF_COLUMN]
        )

        result[self.NAME] = to_categorical(self._mapping, values)
        if hr_value is not None:
            result[self.NAME + '_raw'] = hr_value.encode()

        return result
