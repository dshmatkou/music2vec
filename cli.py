import click
from preprocess_dataset.process_dataset import main as pds
from model.train import main as tr


@click.group()
def cli():
    pass


@cli.command('preprocess_dataset', help='Preprocess dataset')
@click.option('-d', '--dataset-dir', required=True, help='Directory with dataset')
@click.option('-s', '--dataset-size', required=True, help='Size of processing dataset')
@click.option('-a', '--audio-processor', default='mfcc', help='Audio preprocessor')
@click.option('-o', '--output-dir', required=True, help='Output directory')
@click.option('-e', '--with-echonest', required=True, help='Enable echonest features (may reduce result dataset)')
@click.option('-t', '--test-size', default=0.3, type=float, help='How much data will be used in test part of dataset')
def preprocess_dataset(
        dataset_dir,
        dataset_size,
        audio_processor,
        output_dir,
        with_echonest,
        test_size,
):
    pds(
        dataset_dir,
        dataset_size,
        audio_processor,
        output_dir,
        with_echonest,
        test_size,
    )


@cli.command('train', help='Train music2vec model')
@click.option('-d', '--dataset', help='Path to dataset')
def train(dataset):
    tr(dataset)


if __name__ == '__main__':
    cli()
