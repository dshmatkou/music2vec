import argparse
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from audio_processors import get_processor, PROCESSORS


def visualize(matrix):
    print(matrix.shape)
    librosa.display.specshow(matrix, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def parse_args(parser):
    parser.add_argument('--audio-file', help='Audio file to process')
    parser.add_argument(
        '--audio-processor',
        choices=list(PROCESSORS.keys()),
        help='Audio processor',
    )
    args = parser.parse_known_args()
    return args


def main(parser):
    args = parse_args(parser)
    processor = get_processor(args.audio_processor)
    mfcc = processor(args.audio_file)
    visualize(mfcc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
