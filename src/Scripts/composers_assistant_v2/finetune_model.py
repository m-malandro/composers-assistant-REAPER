import torch
import pickle
import time
import constants as cs
import os
import transformers
import nn_training_functions as fn
import functools
import tokenizer_functions as tok

# Instructions: Set EPOCHS, BATCH_SIZE, and CLEANUP_PRETRAIN_DATA below before running this file.
# Set EPOCHS to the range of finetuning epochs that will run when you run this file.
# If EPOCHS = range(0, something),
#   then finetuning will begin from the pretrained model with the most epochs in constants.PATH_TO_MODELS.
# If EPOCHS = range(something > 0, something)
#   then finetuning will resume from the finetuned model with the most epochs in constants.PATH_TO_MODELS / constants.FINETUNE_TASK
# If you are resuming finetune training, and last time EPOCHS was range(x, y),
#   then set EPOCHS to range(y, something > y).
# It is recommended to set dropout_rate = 0.1 in the model's config.json for finetuning.
EPOCHS = range(50, 51)
# BATCH_SIZES[i], i = 0, 1, 2, define batch sizes for "small", "medium", and "large" training examples.
# BATCH_SIZES[i][0] = per-device train batch size; BATCH_SIZES[i][1] = # of gradient accumulation steps
BATCH_SIZES = [(8, 16), (2, 64), (1, 128)]
CLEANUP_FINETUNE_DATA = False  # If True, deletes files created by build_finetune_train_data after they are no longer needed.

CONSTANT_LR = False  # If True, uses Adafactor learning rate of 0.001
N_SLICES_PER_EPOCH = 1  # Saves this many checkpoints per epoch
BF_16 = True
TF_32 = True
# There should be no need to edit anything else in this file.

################################
torch.backends.cuda.matmul.allow_tf32 = TF_32


EPOCHS = list(EPOCHS)
if not EPOCHS:
    raise ValueError('EPOCHS is empty')
TOKENIZER = tok.get_tokenizer()


def get_base_load_path():
    epoch = EPOCHS[0]
    if epoch == 0:
        regime_str = "pretrained_epoch_"
        where = cs.PATH_TO_MODELS
    else:
        regime_str = "finetuned_epoch_"
        where = os.path.join(cs.PATH_TO_MODELS, cs.FINETUNE_TASK)
    pretrained_models = [p for p in os.listdir(where) if p.find(regime_str) == 0]
    pretrained_epochs = []
    for p in pretrained_models:
        p_epoch = p.split(regime_str)[1]
        first_coord = int(p_epoch.split('_')[0])
        if epoch == 0:
            second_coord = 0
        else:
            second_coord = int(p_epoch.split('_')[1])
        pretrained_epochs.append((first_coord, second_coord))
    pretrained_epochs.sort()
    most_recent = pretrained_epochs[-1]
    if epoch == 0:
        base_load_path = os.path.join(where, '{}{}'.format(regime_str, most_recent[0]))
    else:
        base_load_path = os.path.join(where, '{}{}_{}'.format(regime_str, most_recent[0], most_recent[1]))
    return base_load_path


def get_model():
    if not os.path.exists(cs.PATH_TO_MODELS) or len(os.listdir(cs.PATH_TO_MODELS)) == 0:
        raise RuntimeError('No model found to resume training from. Check that {} is nonempty.'.format(cs.PATH_TO_MODELS))

    base_load_path = get_base_load_path()
    print('loading model to finetune from {}'.format(base_load_path))
    M = transformers.T5ForConditionalGeneration.from_pretrained(os.path.join(base_load_path, 'model'))

    M.train()
    return M


def get_optimizer(M):
    if not CONSTANT_LR:
        optimizer = transformers.optimization.Adafactor(M.parameters(),
                                                        scale_parameter=True,
                                                        relative_step=True,
                                                        clip_threshold=1.0,
                                                        warmup_init=True,
                                                        lr=None,
                                                        )
        base_load_path = get_base_load_path()
        optimizer_load_path = os.path.join(base_load_path, 'optimizer')
        with open(os.path.join(optimizer_load_path, 'optimizer_state_dict.txt'), 'rb') as infile:
            # Optimizer states cannot be serialized to json. They must be pickled instead.
            o_sd = pickle.load(infile)
        optimizer.load_state_dict(o_sd)
        lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=1e-3)

    else:
        # constant learning rate
        optimizer = transformers.optimization.Adafactor(M.parameters(),
                                                        scale_parameter=False,
                                                        relative_step=False,
                                                        clip_threshold=1.0,
                                                        lr=None,
                                                        )
        lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=0.001)

    return optimizer, lr_scheduler


def go():
    M = get_model()
    optimizers = get_optimizer(M)

    for epoch in EPOCHS:
        print('Starting epoch {}'.format(epoch))
        for slice_in_epoch in range(0, N_SLICES_PER_EPOCH):
            t0 = time.time()
            dataset = fn.FineTuneTrainDataset(epoch=epoch,
                                              slice_in_epoch=slice_in_epoch,
                                              n_slices_in_epoch=N_SLICES_PER_EPOCH)
            collator = functools.partial(fn.batch_padder, tokenizer=TOKENIZER)

            for subset_i, subset in enumerate(dataset):
                trainer_args = transformers.TrainingArguments(output_dir=r'C:/delete/runs',  # this is not used
                                                              num_train_epochs=1,
                                                              per_device_train_batch_size=BATCH_SIZES[subset_i][0],
                                                              gradient_accumulation_steps=BATCH_SIZES[subset_i][1],
                                                              logging_steps=1,
                                                              save_steps=9999999999999999,  # don't use Trainer's checkpoints
                                                              dataloader_pin_memory=True,
                                                              seed=epoch,
                                                              group_by_length=True,
                                                              tf32=TF_32,
                                                              bf16=BF_16,
                                                              )

                trainer = transformers.Trainer(model=M,
                                               train_dataset=subset,
                                               data_collator=collator,
                                               optimizers=optimizers,
                                               args=trainer_args)

                trainer.train()

            # after training, save model
            base_path = os.path.join(cs.PATH_TO_MODELS, cs.FINETUNE_TASK, 'finetuned_epoch_{}_{}'.format(epoch, slice_in_epoch))
            M.save_pretrained(save_directory=os.path.join(base_path, 'model'))

            # also save optimizer save state
            sd = optimizers[0].state_dict()
            optimizer_save_path = os.path.join(base_path, 'optimizer')
            if not os.path.exists(optimizer_save_path):
                os.mkdir(optimizer_save_path)
            with open(os.path.join(optimizer_save_path, 'optimizer_state_dict.txt'), 'wb') as outfile:
                # Optimizer states cannot be serialized to json. They must be pickled instead.
                pickle.dump(sd, outfile)

            print('Model saved. Finetune task = {}, epoch = {}, slice = {} COMPLETE in {} sec.'.format(cs.FINETUNE_TASK, epoch, slice_in_epoch, time.time() - t0))

        if CLEANUP_FINETUNE_DATA:
            print('Removing finetune data for epoch {}'.format(epoch))
            for folder, _, fnames in os.walk(os.path.join(cs.PATH_TO_TEMP_FILES, cs.FINETUNE_TASK)):
                for fname in fnames:
                    if fname.find('finetune_epoch_{}_'.format(epoch)) == 0:
                        del_file = os.path.join(folder, fname)
                        os.remove(del_file)


if __name__ == '__main__':
    go()
