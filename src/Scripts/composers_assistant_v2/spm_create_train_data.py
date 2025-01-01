import json
import os.path
import time
import spm_train_functions as fn
import constants as cs
from multiprocessing import Pool
import math

if __name__ == '__main__':
    path = cs.PATH_TO_PROCESSED_TRAIN_MIDI
    t_st = time.time()
    print('counting songs...')
    t0 = time.time()
    song_count = fn.get_song_count(path=path)
    print('Found {} songs in {} sec'.format(song_count, time.time() - t0))
    print('Creating examples to train Sentencepiece model...')

    n_examples_per_song_get = round(3*cs.SPM_NUM_EXAMPLES/song_count)
    n_examples_per_song_wanted = math.ceil(cs.SPM_NUM_EXAMPLES/song_count)
    n_tries_per_song = 5 * n_examples_per_song_get

    P = Pool()

    examples_including_note_offs = set()
    examples_excluding_note_offs = set()
    examples_using_note_lengths = set()
    examples_using_note_durations = set()
    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            print('loading file {}'.format(fname))
            t0 = time.time()
            with open(os.path.join(path, fname)) as infile:
                d = json.load(infile)
            print('file loaded in {} sec'.format(time.time()-t0))

            inputs = [(d[p], n_examples_per_song_get, n_tries_per_song) for p in d]

            for i, res in enumerate(P.imap_unordered(fn.create_spm_examples_parallel, inputs, chunksize=10)):
                st_inc = len(examples_including_note_offs)
                st_exc = len(examples_excluding_note_offs)
                st_len = len(examples_using_note_lengths)
                st_dur = len(examples_using_note_durations)
                for ex in res['examples_including_note_offs']:
                    examples_including_note_offs.add(ex)
                    if len(examples_including_note_offs) - st_inc > n_examples_per_song_wanted:
                        break
                for ex in res['examples_excluding_note_offs']:
                    examples_excluding_note_offs.add(ex)
                    if len(examples_excluding_note_offs) - st_exc > n_examples_per_song_wanted:
                        break
                for ex in res['examples_using_note_lengths']:
                    examples_using_note_lengths.add(ex)
                    if len(examples_using_note_lengths) - st_len > n_examples_per_song_wanted:
                        break
                for ex in res['examples_using_note_duration_commands']:
                    examples_using_note_durations.add(ex)
                    if len(examples_using_note_durations) - st_dur > n_examples_per_song_wanted:
                        break

                if (i+1) % 100 == 0:
                    print(i+1, 'songs in this file processed so far')

    print('Created {} examples that include note offs. Writing to file...'.format(len(examples_including_note_offs)))
    with open('spm_train_incl_note_offs.txt', 'w') as outfile:
        for ex in examples_including_note_offs:
            outfile.write(ex + '\n')

    print('Created {} examples that exclude note offs. Writing to file...'.format(len(examples_excluding_note_offs)))
    with open('spm_train_excl_note_offs.txt', 'w') as outfile:
        for ex in examples_excluding_note_offs:
            outfile.write(ex + '\n')

    print('Created {} examples that use note lengths. Writing to file...'.format(len(examples_using_note_lengths)))
    with open('spm_train_note_lengths.txt', 'w') as outfile:
        for ex in examples_using_note_lengths:
            outfile.write(ex + '\n')

    print('Created {} examples that use note duration commands. Writing to file...'.format(len(examples_using_note_durations)))
    with open('spm_train_note_durations.txt', 'w') as outfile:
        for ex in examples_using_note_durations:
            outfile.write(ex + '\n')

    print('All done in {} sec'.format(time.time() - t_st))
