import copy
import os
import json
import random

import constants as cs
import encoding_functions as enc
import preprocessing_functions as pre
import midisong as ms


def get_song_count(path):
    song_count = 0
    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            song_count += len(d)
    return song_count


def create_spm_examples(midisongbymeasure_dict, n_examples=10, n_tries=1000):
    S_orig = pre.midisongbymeasure_from_save_dict(midisongbymeasure_dict)
    matrices_incl = {}
    matrices_excl = {}
    matrices_len = {}
    matrices_dur = {}

    for trans_amt in range(cs.AUG_TRANS_MIN, cs.AUG_TRANS_MAX + 1):
        S = copy.copy(S_orig)
        S.transpose(amt=trans_amt)

        # Note: This also cleans up any note duplicates at the same click that result
        enc.transpose_into_acceptable_ranges_TT(S)

        S.sort_tracks_by_inst_and_avg_note_pitch()
        for t in S.tracks:
            t.sort()

        matrices_incl[trans_amt] = enc.get_string_encoding_matrices(S, note_off_treatment='include')
        matrices_excl[trans_amt] = enc.get_string_encoding_matrices(S, note_off_treatment='exclude')
        matrices_len[trans_amt] = enc.get_string_encoding_matrices(S, note_off_treatment='length')
        matrices_dur[trans_amt] = enc.get_string_encoding_matrices(S, note_off_treatment='duration')

    tries_so_far = 0
    examples_incl = []
    examples_excl = []
    examples_len = []
    examples_dur = []
    res = {}

    while tries_so_far < n_tries:
        t = random.randint(cs.AUG_TRANS_MIN, cs.AUG_TRANS_MAX)
        measure_i = random.randint(0, S_orig.get_n_measures())
        tr_i = random.randint(0, len(S_orig.tracks))

        heads, tails = matrices_incl[t]
        example = tails[(tr_i, measure_i)]
        if len(example) > 30 and example not in examples_incl:
            examples_incl.append(example)

        heads, tails = matrices_excl[t]
        example = tails[(tr_i, measure_i)]
        if len(example) > 30 and example not in examples_excl:
            examples_excl.append(example)

        heads, tails = matrices_len[t]
        example = tails[(tr_i, measure_i)]
        if len(example) > 30 and example not in examples_len:
            examples_len.append(example)

        heads, tails = matrices_dur[t]
        example = tails[(tr_i, measure_i)]
        if len(example) > 30 and example not in examples_dur:
            examples_dur.append(example)

        if len(examples_incl) >= n_examples and \
                len(examples_excl) >= n_examples and \
                len(examples_len) >= n_examples and \
                len(examples_dur) >= n_examples:
            tries_so_far = n_tries  # to break out of the while loop

        tries_so_far += 1

    res['examples_including_note_offs'] = examples_incl
    res['examples_excluding_note_offs'] = examples_excl
    res['examples_using_note_lengths'] = examples_len
    res['examples_using_note_duration_commands'] = examples_dur
    return res


def create_spm_examples_parallel(T):
    return create_spm_examples(*T)


def get_user_defined_symbols():
    # user_defined_symbols = [';M']
    user_defined_symbols = []
    for x in range(8):
        # BPM
        user_defined_symbols.append(';B:{}'.format(x))

    for x in range(8):
        # "loudness" level of the measure
        # 0 = ppp
        # 1 = pp
        # 2 = p
        # 3 = mp
        # 4 = mf
        # 5 = f
        # 6 = ff
        # 7 = fff
        user_defined_symbols.append(';M:{}'.format(x))

    for x in range(1, 8 * ms.extended_lcm(cs.QUANTIZE) + 1):
        user_defined_symbols.append(';L:{}'.format(x))

    for x in range(258):
        user_defined_symbols.append(';I:{}'.format(x))

    # Up to 64 repeated tracks of the same inst, I guess. Max is about 30 in train data.
    # Most of these tokens will be significantly undertrained using my dataset, but that's ok.
    for x in range(1, 64):
        user_defined_symbols.append(';R:{}'.format(x))

    for x in range(256):
        user_defined_symbols.append(';<extra_id_{}>'.format(x))

    user_defined_symbols.append(';<mono>')
    user_defined_symbols.append(';<poly>')
    # user_defined_symbols.append(';<rhythm_copy>')
    # user_defined_symbols.append(';<rhythm_new>')

    for x in range(512):
        user_defined_symbols.append(';<instruction_{}>'.format(x))

    return user_defined_symbols
