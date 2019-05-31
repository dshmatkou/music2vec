import csv
import numpy as np
import tensorflow as tf
from model.model import model_fn
from preprocess_dataset.audio.processors import get_processor

PROCESSOR = 'mfcc'


def load_genres(genres_metadata_fn):
    with open(genres_metadata_fn, 'r') as f:
        reader = csv.DictReader(f)
        items = [
            (int(item['genre_id']), item['title'])
            for item in reader
        ]
        return items


def extract_genres(vector, genres):
    result = [
        (genre_title, vector[genre_id])
        for genre_id, genre_title in genres
    ]
    result = sorted(result, key=lambda x: x[1])
    return result


def demonstrate(music_file, model_path, genres_metadata_fn):
    processor = get_processor(PROCESSOR)
    genres = load_genres(genres_metadata_fn)
    features = processor(music_file)
    features = np.reshape(features, (1,) + features.shape + (1,))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=10,
            model_dir=model_path,
        )
    )

    predictions = estimator.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'feature': features},
            shuffle=False
        )
    )

    pred = next(predictions)
    result = extract_genres(pred['genres_top'], genres)
    for genre_title, prob in result:
        print(genre_title, ':', prob)
