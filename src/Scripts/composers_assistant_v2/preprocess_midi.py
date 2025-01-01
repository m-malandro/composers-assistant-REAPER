import multiprocessing
import os
import time
import constants
import preprocessing_functions as pre
import json


# LMD takes about 9.5 hours on 4 cores at home
if __name__ == '__main__':
    __file__ = 'preprocess_midi.py'
    P = multiprocessing.Pool()

    FOLDERS = [(constants.PATH_TO_TRAIN_MIDI, constants.PATH_TO_PROCESSED_TRAIN_MIDI)]

    if constants.PATH_TO_VAL_MIDI:
        FOLDERS.insert(0, (constants.PATH_TO_VAL_MIDI, constants.PATH_TO_PROCESSED_VAL_MIDI))

    if constants.PATH_TO_TEST_MIDI:
        FOLDERS.append((constants.PATH_TO_TEST_MIDI, constants.PATH_TO_PROCESSED_TEST_MIDI))

    for T in FOLDERS:
        base_folder, out_folder = T

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        print('processing {}'.format(base_folder))
        t0 = time.time()

        paths = []
        for folder, _, fnames in os.walk(base_folder):
            for fname in fnames:
                if '.mid' in fname.lower():
                    paths.append(os.path.join(folder, fname))
        print(len(paths), 'files to process')

        chunksize = 10000
        cur_chunk = 0

        while cur_chunk * chunksize < len(paths):
            cur_dump_dict = {}
            st = cur_chunk * chunksize
            end = (cur_chunk + 1) * chunksize
            for i, res in enumerate(P.imap_unordered(pre.preprocess_midi_to_save_dict, paths[st: end], chunksize=10)):
                p, d = res
                if d is not None:
                    cur_dump_dict[p] = d

                if (i + 1) % 1000 == 0:
                    print(i + 1, 'files in chunk {} done so far'.format(cur_chunk))

            print('saving chunk {}'.format(cur_chunk))
            with open(os.path.join(out_folder, '{}.txt'.format(cur_chunk)), 'w') as outfile:
                json.dump(cur_dump_dict, outfile)

            cur_chunk += 1

        print('{} processed in {} sec'.format(base_folder, time.time() - t0))

    print('all done!')
