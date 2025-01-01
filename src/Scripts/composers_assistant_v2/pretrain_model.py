import torch
import pickle
import time
import constants as cs
import os
import transformers
import nn_training_functions as fn
import functools
import tokenizer_functions as tok

# Instructions: Set EPOCHS, BATCH_SIZE, and CLEANUP_PRETRAIN_DATA before running this file.
# Set EPOCHS to the range of epochs that you will pretrain when you run this file. If you are pretraining from scratch,
# then---and only then---should you set EPOCHS to range(0, something).
# If you are resuming training, and last time EPOCHS was range(x, y), then set EPOCHS to range(y, something > y).
EPOCHS = range(0, 2)
BATCH_SIZE = (32, 2)  # BATCH_SIZE[0] = per-device train batch size; BATCH_SIZE[1] = # of gradient accumulation steps
CLEANUP_PRETRAIN_DATA = False  # If True, deletes files created by build_pretrain_data after they are no longer needed.
BF_16 = True
TF_32 = True

# There should be no need to edit anything else in this file, unless you are pretraining from scratch and
# want to alter something about the model or the way it is trained

torch.backends.cuda.matmul.allow_tf32 = TF_32

EPOCHS = list(EPOCHS)
if not EPOCHS:
    raise ValueError('EPOCHS is empty')
TOKENIZER = tok.get_tokenizer()


def get_base_load_path():
    if EPOCHS[0] == 0:
        return ''
    else:
        base_load_path = os.path.join(cs.PATH_TO_MODELS, 'pretrained_epoch_{}'.format(EPOCHS[0] - 1))
        return base_load_path


def get_optimizer(M):
    optimizer = transformers.optimization.Adafactor(M.parameters(),
                                                    scale_parameter=True,
                                                    relative_step=True,
                                                    clip_threshold=1.0,
                                                    warmup_init=True,
                                                    lr=None,
                                                    # use weight_decay = 0 for the first two epochs, at minimum
                                                    weight_decay=0.00)

    base_load_path = get_base_load_path()
    if base_load_path:
        optimizer_load_path = os.path.join(base_load_path, 'optimizer')
        with open(os.path.join(optimizer_load_path, 'optimizer_state_dict.txt'), 'rb') as infile:
            # Optimizer states cannot be serialized to json. They must be pickled instead.
            o_sd = pickle.load(infile)
        optimizer.load_state_dict(o_sd)
    # If you want to change the weight decay setting, you need to do it after this line
    # e.g., (note: untested):
    # optimizer.state_dict()['param_groups'][0]['weight_decay'] = 1e^-6

    # initial_lr is ignored b/c we're using scale_parameter and relative_step
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=1e-3)

    return optimizer, lr_scheduler


def get_model():
    # if path to models is empty or has nothing in it, then we need to create a new model
    if not os.path.exists(cs.PATH_TO_MODELS) or len(os.listdir(cs.PATH_TO_MODELS)) == 0 or EPOCHS[0] == 0:
        model_config = transformers.T5Config(vocab_size=TOKENIZER.vocab_size(),
                                             d_model=cs.D_MODEL,
                                             d_kv=cs.D_MODEL // cs.N_HEADS,
                                             d_ff=4 * cs.D_MODEL,
                                             num_layers=cs.N_LAYERS,
                                             num_heads=cs.N_HEADS,
                                             relative_attention_num_buckets=4096,
                                             relative_attention_max_distance=4096,
                                             dropout_rate=0.0,
                                             feed_forward_proj='gated-gelu',
                                             use_cache=False,
                                             pad_token_id=TOKENIZER.pad_id(),
                                             eos_token_id=TOKENIZER.eos_id(),
                                             bos_token_id=TOKENIZER.bos_id(),
                                             decoder_start_token_id=TOKENIZER.pad_id())

        M = transformers.T5ForConditionalGeneration(model_config)
        print('Created new T5 model with {} parameters to train'.format(M.num_parameters()))

    else:
        # load model and optimizer for training
        base_load_path = get_base_load_path()
        print(f'Loading pretrained model from {base_load_path} to resume pretraining')
        M = transformers.T5ForConditionalGeneration.from_pretrained(os.path.join(base_load_path, 'model'))

    M.train()  # sets M to training mode. Doesn't actually train M.
    return M


def go():
    M = get_model()
    optimizers = get_optimizer(M)

    for epoch in EPOCHS:
        print('Starting epoch {}'.format(epoch))
        t0 = time.time()
        dataset = fn.PreTrainDataset(epoch=epoch, mode='train')
        collator = functools.partial(fn.batch_padder, tokenizer=TOKENIZER)

        trainer_args = transformers.TrainingArguments(output_dir=r'C:/delete/runs',  # this is not used
                                                      num_train_epochs=1,
                                                      per_device_train_batch_size=BATCH_SIZE[0],
                                                      gradient_accumulation_steps=BATCH_SIZE[1],
                                                      logging_steps=10,
                                                      save_steps=9999999999999999,  # don't use Trainer's checkpoints
                                                      dataloader_pin_memory=True,
                                                      seed=epoch,
                                                      group_by_length=True,
                                                      tf32=TF_32,
                                                      bf16=BF_16,
                                                      )

        trainer = transformers.Trainer(model=M,
                                       train_dataset=dataset,
                                       data_collator=collator,
                                       optimizers=optimizers,
                                       args=trainer_args)

        trainer.train()

        # after training, save model
        base_path = os.path.join(cs.PATH_TO_MODELS, 'pretrained_epoch_{}'.format(epoch))
        M.save_pretrained(save_directory=os.path.join(base_path, 'model'))

        # after training, save optimizer state
        sd = optimizers[0].state_dict()
        optimizer_save_path = os.path.join(base_path, 'optimizer')
        if not os.path.exists(optimizer_save_path):
            os.mkdir(optimizer_save_path)
        with open(os.path.join(optimizer_save_path, 'optimizer_state_dict.txt'), 'wb') as outfile:
            # Optimizer states cannot be serialized to json. They must be pickled instead.
            pickle.dump(sd, outfile)

        print('Model and optimizer state saved. Epoch {} COMPLETE in {} sec.'.format(epoch, time.time() - t0))

        if CLEANUP_PRETRAIN_DATA:
            print('Removing pretrain data for epoch {}'.format(epoch))
            for folder, _, fnames in os.walk(cs.PATH_TO_TEMP_FILES):
                for fname in fnames:
                    if fname.find('pretrain_epoch_{}_'.format(epoch)) == 0:
                        del_file = os.path.join(folder, fname)
                        os.remove(del_file)

    print('All done.')


if __name__ == '__main__':
    go()
