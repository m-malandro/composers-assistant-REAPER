import nn_training_functions as fn
from multiprocessing import Pool
import os
import constants as cs
import tokenizer_functions as tok

TOKENIZER = tok.get_tokenizer()

if __name__ == '__main__':
    __file__ = 'build_val_and_test_finetune_data_infill.py'

    P = Pool()

    if not os.path.exists(cs.PATH_TO_TEMP_FILES):
        os.mkdir(cs.PATH_TO_TEMP_FILES)

    for mode in ['test']:
        for n_measures in (8, ):
            for mask_pattern_type in ("0half", "1singleinst", "2last"):
                fn.build_val_test_finetune_data_infill(tokenizer=TOKENIZER, epoch=0, pool=P, mode=mode,
                                                       mask_pattern_type=mask_pattern_type,
                                                       n_measures=n_measures)

