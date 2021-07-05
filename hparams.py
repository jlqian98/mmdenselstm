import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # input Parameters        #
        ################################
        num_classes = 4,
        SR = 10880,
        H = 512,
        fft_size = 1022,
        patch_length= 256, 
        training_rate = 0.95,
        insts = ['drums', 'bass', 'other', 'vocals'],
        out_channels_list = [0, 1, 2, 3],    # 输出channel个数

        ################################
        # Training Parameters        #
        ################################
        beta1 = 0.1,
        beta2 = 0.999,
        batch_size = 6,
        num_workers = 4,
        num_epoch = 1000,

        lr = 0.001,
        warm_start = False,
        checkpoint = None,
        log_step = 50,
        save_iter = 20,
        sdr_iter = 50,
        patience = 20,
        device='cuda',

        ################################
        # Dataset        #
        ################################
        dataset_dir = "/home/jlqian/dataset/musdb18/musdb18",
        wave_path = "/media/jlqian/data/musdb18_wav",
        fft_path = "/home/jlqian/dataset/musdb18/musdb18_fft_augmented_remix_10880_max",
        separate_path = './separate',
        eval_path = './eval',
        test_wav_estimate = './test_estimate',

        model_name = 'mmdenselstm',
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
