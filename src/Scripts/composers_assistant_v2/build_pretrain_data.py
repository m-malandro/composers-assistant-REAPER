import nn_training_functions as fn
from multiprocessing import Pool
import os
import constants as cs
import tokenizer_functions as tok

# Instructions: Edit range(x, y) to set the epochs for which you're building pretrain data. Then run this file.
# If you are just starting the pretraining process, begin with epoch 0.
EPOCHS = range(0, 6)

TOKENIZER = tok.get_tokenizer()

if __name__ == '__main__':
    P = Pool()

    if not os.path.exists(cs.PATH_TO_TEMP_FILES):
        os.mkdir(cs.PATH_TO_TEMP_FILES)

    # first, build val data if necessary
    if not os.path.exists(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_short.txt')):
        fn.build_pretrain_data(tokenizer=TOKENIZER, epoch=0, pool=P, mode='val_short')
    if not os.path.exists(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_long.txt')):
        fn.build_pretrain_data(tokenizer=TOKENIZER, epoch=0, pool=P, mode='val_long')

    # then build train data
    for epoch in EPOCHS:
        fn.build_pretrain_data(tokenizer=TOKENIZER, epoch=epoch, pool=P, mode='train')

    print('All done')
