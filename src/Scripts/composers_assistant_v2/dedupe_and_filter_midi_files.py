import os

from midisong import *
import dedupe_and_filter_midi_files_functions as dd
from multiprocessing import Pool


dedupe_dir = r'C:\delete\lakh'

dedupe_seen_so_far = set()
passed = []
failed = []
dedupe_str_to_path = {}


def process_dict(d, passed_L, failed_L, dedupe_seen_so_far_set, dedupe_str_to_path_dict):
    if d['error']:
        failed_L.append((d['p'], 'error'))
        return

    if d['n_measures'] < 4:
        failed_L.append((d['p'], 'n_measures: {}'.format(d['n_measures'])))
        return

    if d['cos_sim'] > 0.8:
        failed_L.append((d['p'], 'cos_sim: {}'.format(d['cos_sim'])))
        return

    for s in d['dedupe_strs']:
        if s in dedupe_seen_so_far_set:
            match_p = dedupe_str_to_path_dict[s]
            failed_L.append((d['p'], 'matches ' + match_p))
            return

    # if we make it here, then this song has passed all of our filtering and deduping tests
    passed_L.append(d['p'])
    dedupe_seen_so_far_set.add(d['dedupe_strs'][0])
    dedupe_str_to_path_dict[d['dedupe_strs'][0]] = d['p']
    return


if __name__ == '__main__':
    paths = []
    for folder, _, fnames in os.walk(dedupe_dir):
        for fname in fnames:
            if '.mid' in fname.lower():
                p = os.path.join(folder, fname)
                paths.append(p)
    print('paths created', len(paths))
    P = Pool()
    for i, res in enumerate(P.imap_unordered(dd.get_deduping_and_filtering_info, paths, chunksize=10)):
        process_dict(res, passed_L=passed, failed_L=failed, dedupe_seen_so_far_set=dedupe_seen_so_far,
                     dedupe_str_to_path_dict=dedupe_str_to_path)
        if (i + 1) % 100 == 0:
            print(i + 1, 'done so far')


    import pickle
    with open('zzzzz_failed.txt', 'wb') as outfile:
        pickle.dump(failed, outfile)
