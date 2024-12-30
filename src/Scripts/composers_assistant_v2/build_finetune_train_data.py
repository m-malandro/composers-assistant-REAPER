import nn_training_functions as fn
from multiprocessing import Pool
import os
import constants as cs
import tokenizer_functions as tok

# Instructions: Edit range(x, y) to set the epochs for which you're building pretrain data. Then run this file.
# If you are just starting the finetuning process, begin with epoch 0.
# The default on the following line is how you generate one epoch of fine-tuning data for personalization of the included model.
EPOCHS = range(50, 51)
FORCE_N_MEASURES = False  # Forces this many measures in each training example. Set to False to allow for many different numbers of measures.
N_CPUS = 24  # Adjust to be <= the number of logical cores on your device.

TOKENIZER = tok.get_tokenizer()

if __name__ == '__main__':
    __file__ = 'build_finetune_train_data.py'
    P = Pool(N_CPUS)

    if not os.path.exists(cs.PATH_TO_TEMP_FILES):
        os.mkdir(cs.PATH_TO_TEMP_FILES)

    path = cs.PATH_TO_PROCESSED_TRAIN_MIDI
    for epoch in EPOCHS:
        fn.build_finetune_train_data(tokenizer=TOKENIZER, epoch=epoch, pool=P, path=path, force_n_measures=FORCE_N_MEASURES)

    print('All done')
