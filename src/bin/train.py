import sys
import argparse
import os
import multiprocessing
from datetime import datetime
from keras import callbacks

if './' not in sys.path:
    sys.path.append('./')

from src.utils.config import get_config
from src.generators.celeb_a import CelebAGenerator
from src.models.celeb_a import CelebAModel


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('config', help='The path to the config file')
    return parser.parse_args(args)

def decide_parallelization():
    # Use multiprocessing only on Linux
    use_multiprocessing = not os.name == 'nt'
    workers = multiprocessing.cpu_count() if use_multiprocessing else 1
    workers = min(workers, 4)
    return use_multiprocessing, workers

def create_train_directory(save_folder_root):
    # Create new folder with current date
    date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    save_folder = os.path.join(save_folder_root, date)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder

def get_callbacks(save_folder, save_frequency):
    # Define save path
    save_path = os.path.join(save_folder, '{epoch}.h5')
    log_path = os.path.join(save_folder, 'training_log.csv')

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(save_path, period=save_frequency)
    logger = callbacks.CSVLogger(log_path)
    lr_reduce = callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)

    return [checkpoint, logger, lr_reduce]

def run(conf):
    
    # Create generators
    train_gen = CelebAGenerator.from_conf(conf, is_train=True)
    test_gen = CelebAGenerator.from_conf(conf, is_train=False)

    # Create model controller
    model_controller = CelebAModel.from_conf(conf)

    # Determine parallel processing based on operating system
    use_multiprocessing, workers = decide_parallelization()

    # Get new training directory and save weights before training
    save_folder = create_train_directory(conf.path.models_root)

    print("\nUNFREEZING LAYERS\n")

    # Unfreeze, create callbacks and train again
    model_controller.unfreeze()
    model_controller.compile()
    callbacks = get_callbacks(save_folder, conf.train.save_frequency)
    model_controller.model.fit_generator(
        train_gen,
        epochs=conf.train.epochs,
        validation_data=test_gen,
        callbacks=callbacks,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    conf = get_config(args.config)
    run(conf)


if __name__ == '__main__':
    main()