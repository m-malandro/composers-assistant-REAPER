import os.path

import sentencepiece as spm
import constants as cs
import spm_train_functions
import time
import shutil

t0 = time.time()

user_defined_symbols = spm_train_functions.get_user_defined_symbols()

spm_options = {'input_format': 'text',
               'model_type': 'unigram',
               'character_coverage': 1,
               'vocab_size': cs.N_PHRASES_FOR_VOCAB + len(user_defined_symbols),
               'split_by_number': False,
               'split_by_unicode_script': False,
               'split_by_whitespace': False,
               'split_digits': False,
               'user_defined_symbols': user_defined_symbols,
               'max_sentencepiece_length': 96,
               'max_sentence_length': 100000,  # bytes
               'shuffle_input_sentence': True,
               'add_dummy_prefix': False,
               'unk_id': 0,
               'bos_id': 1,
               'eos_id': 2,
               'pad_id': 3,
               'unk_surface': '<unk>'
               }

if cs.SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS:
    spm.SentencePieceTrainer.Train(input=['spm_train_note_durations.txt'],
                                   model_prefix='spm_incl_note_duration_commands',
                                   **spm_options)
    target_dir = os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm')
    os.makedirs(target_dir, exist_ok=True)
    shutil.move('spm_incl_note_duration_commands.model',
                os.path.join(target_dir, 'spm_incl_note_duration_commands.model'))
    shutil.move('spm_incl_note_duration_commands.vocab',
                os.path.join(target_dir, 'spm_incl_note_duration_commands.vocab'))
    # os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm'...

if cs.SPM_TRAIN_MODEL_WITH_NOTE_OFFS:
    spm.SentencePieceTrainer.Train(input=['spm_train_incl_note_offs.txt'],
                                   model_prefix='spm_incl_note_offs',
                                   **spm_options
                                   )
    target_dir = os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm')
    os.makedirs(target_dir, exist_ok=True)
    shutil.move('spm_incl_note_offs.model',
                os.path.join(target_dir, 'spm_incl_note_offs.model'))
    shutil.move('spm_incl_note_offs.vocab',
                os.path.join(target_dir, 'spm_incl_note_offs.vocab'))

if cs.SPM_TRAIN_MODEL_WITHOUT_NOTE_OFFS:
    spm.SentencePieceTrainer.Train(input=['spm_train_excl_note_offs.txt'],
                                   model_prefix='spm_excl_note_offs',
                                   **spm_options
                                   )
    target_dir = os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm')
    os.makedirs(target_dir, exist_ok=True)
    shutil.move('spm_excl_note_offs.model',
                os.path.join(target_dir, 'spm_excl_note_offs.model'))
    shutil.move('spm_excl_note_offs.vocab',
                os.path.join(target_dir, 'spm_excl_note_offs.vocab'))

if cs.SPM_TRAIN_MODEL_WITH_NOTE_LENGTHS:
    spm.SentencePieceTrainer.Train(input=['spm_train_note_lengths.txt'],
                                   model_prefix='spm_incl_note_lengths',
                                   **spm_options)
    target_dir = os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm')
    os.makedirs(target_dir, exist_ok=True)
    shutil.move('spm_incl_note_lengths.model',
                os.path.join(target_dir, 'spm_incl_note_lengths.model'))
    shutil.move('spm_incl_note_lengths.vocab',
                os.path.join(target_dir, 'spm_incl_note_lengths.vocab'))


print('done training spm models in {} sec'.format(time.time() - t0))
