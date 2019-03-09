from preprocess_dataset.audio_processors.mfcc import process_file

PROCESSORS = {
    'mfcc': process_file,
}


def get_processor(processor_name):
    return PROCESSORS[processor_name]
