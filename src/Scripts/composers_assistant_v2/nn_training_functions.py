import copy
import json
import math
import pickle
import random
import time
import bisect

import constants as cs
import midisong as ms
import encoding_functions as enc
import preprocessing_functions as pre
import os
import torch
import collections
from tokenizer_functions import spm_type_to_note_off_treatment

N_FINETUNE_EXAMPLES_MULTIPLIER = 30

TRAIN_ON_RANDOM_PERMUTATIONS_OF_MASKS = True

# some augmentation probabilities
P_4_4_to_8_4 = 0.03
P_4_4_to_1_4 = 0.01
P_4_4_to_2_4 = 0.01


def finetune_sampling_prob(S: "ms.MidiSongByMeasure", p: "str" = ''):
    """p the path to the midi file that created S"""
    # example: p = 'PATH_TO_TRAIN_MIDI\permissive-midi-CC0\12-SenseZeraMIDI\01 - Terra Serafina.mid'

    disliked_paths = []
    if any(x in p for x in disliked_paths):
        print(f'prob = 0.0; skipping {p}')
        return 0.0

    path_multiplier = 1.0

    if len(S.tracks) == 1:
        return path_multiplier * 0.125
    elif len(S.tracks) == 2:
        return path_multiplier * 0.65
    elif len(S.tracks) == 3:
        return path_multiplier * 0.875
    return path_multiplier * 1.0


def get_finetune_examples_multiplier(S: ms.MidiSongByMeasure, p: str = ''):
    # you can write code here to re-weight sampling frequencies
    return N_FINETUNE_EXAMPLES_MULTIPLIER


def passes_final_finetune_example_check(input_ids_str: "str", labels_str: "str", tokenizer):
    # if "I:128;<extra_id" in input_ids_str:  # drum-writing examples only
    #     return True
    # else:
    #     return False

    return True


def is_good_for_val_test_infill(S: ms.MidiSongByMeasure,
                                tokenizer,
                                mask_pattern_type_str: str,
                                mask: set[tuple[int, int]],
                                unmasked_tr_measures: set[tuple[int, int]],
                                measure_st: int,
                                n_measures: int):
    # only support <= 256 masks
    if len(mask) > 256:
        return False
    # want at least 2 masked tr-measures
    if len(mask) < 2:
        return False
    # need at least one unmasked tr-measure
    if len(unmasked_tr_measures) < 1:
        return False

    # for this pattern type, make sure we are filling in at least e.g. 7 measures for an 8-measure example
    if mask_pattern_type_str == '1singleinst':
        if len(mask) < n_measures - 1:
            return False

    # make sure the biggest possible thing will fit in memory
    s, labels = val_test_infill_encode(S=S,
                                       mask_locations=list(mask),
                                       measure_slice=(measure_st, measure_st + n_measures),
                                       include_no_octave_shift_instructions=True,
                                       include_hi_lo_note_instructions_per_track_measure=True,
                                       do_rhythm_conditioning=True,
                                       rhythmic_conditioning_type='n_pitch_classes_and_n_notes',
                                       return_labels_too=True)

    if len(tokenizer.encode(s)) > cs.MAX_LEN:
        return False
    if len(tokenizer.encode(labels)) > cs.MAX_LEN:
        return False

    return True


def val_test_infill_encode(S: ms.MidiSongByMeasure,
                           mask_locations: list[tuple[int, int]],
                           measure_slice: tuple[int, int],
                           include_no_octave_shift_instructions: bool,
                           include_hi_lo_note_instructions_per_track_measure: bool,
                           do_rhythm_conditioning: bool,
                           rhythmic_conditioning_type: str or None,
                           return_labels_too: bool,
                           include_vert_options: bool = True,
                           include_horiz_density: bool = True,
                           include_other_horiz_options: bool = True,
                           include_pitch_step_leap_probs: bool = True,
                           include_hi_lo_note_instructions_per_track: bool = True,
                           ):
    track_measure_commands = collections.defaultdict(str)

    if include_no_octave_shift_instructions:
        for T in mask_locations:
            tr_i, m_i = T
            if not S.is_octave_collapse_of_some_track_in_this_measure(tr_i=tr_i, measure_i=m_i):
                track_measure_commands[T] += enc.instruction_str(1, enc.MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE)

    if include_hi_lo_note_instructions_per_track_measure:
        for T in mask_locations:
            tr_i, measure_i = T
            pitch_range = S.pitch_range(tr_i=tr_i, measures=[measure_i])
            if pitch_range is not None:
                lo, hi = pitch_range
                is_drum = S.tracks[tr_i].is_drum
                # strict instructions only
                instruction_lo = enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT, is_drum=is_drum)
                instruction_hi = enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT, is_drum=is_drum)
                track_measure_commands[T] += instruction_lo
                track_measure_commands[T] += instruction_hi

    if do_rhythm_conditioning:
        rhy_cond_locs = mask_locations
    else:
        rhy_cond_locs = set()

    commands_at_end = collections.defaultdict(str)

    # for commands at end, precompute some information
    masked_measure_indexes_by_track_index = collections.defaultdict(set)
    for T in mask_locations:
        masked_measure_indexes_by_track_index[T[0]].add(T[1])

    masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index = collections.defaultdict(set)
    for T in rhy_cond_locs:
        masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index[T[0]].add(T[1])

    masked_measure_indexes_without_rhythmic_conditioning_by_track_index = collections.defaultdict(set)
    for tr_i, MIs in masked_measure_indexes_by_track_index.items():
        ERs = masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index[tr_i]
        masked_measure_indexes_without_rhythmic_conditioning_by_track_index[tr_i] = MIs.difference(ERs)

    if include_vert_options:
        # vert density
        if rhythmic_conditioning_type in (None, '1d_flattening'):
            mask_dict_to_use_for_vert_density = masked_measure_indexes_by_track_index
        elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
            mask_dict_to_use_for_vert_density = masked_measure_indexes_without_rhythmic_conditioning_by_track_index
        else:
            raise ValueError(f'rhythmic_conditioning_type={rhythmic_conditioning_type} not understood')
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=mask_dict_to_use_for_vert_density,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_VERT_NOTE_ONSET_DENSITY)

        # vert note onset n pitch classes
        if rhythmic_conditioning_type in (None, '1d_flattening'):
            mask_dict_to_use_for_vert_n_pitch_classes_avg = masked_measure_indexes_by_track_index
        elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
            mask_dict_to_use_for_vert_n_pitch_classes_avg = masked_measure_indexes_without_rhythmic_conditioning_by_track_index
        else:
            raise ValueError(f'rhythmic_conditioning_type={rhythmic_conditioning_type} not understood')
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=mask_dict_to_use_for_vert_n_pitch_classes_avg,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG)

    if include_horiz_density:
        # horiz density
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY)

    if include_other_horiz_options:
        # horiz note onset diversity percentage
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE)

        # horiz note onset irregularity
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY)

    if include_pitch_step_leap_probs:
        # pitch step and leap prob
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_by_track_index,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_PITCH_STEP_PROB)
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_by_track_index,
                                include_commands_at_end='all',
                                measurement_str=enc.MEASUREMENT_PITCH_LEAP_PROB)

    if include_hi_lo_note_instructions_per_track:
        # highest note and lowest note: only do this if we're not already doing it by track measure
        if not include_hi_lo_note_instructions_per_track_measure:
            for tr_i, MIs in masked_measure_indexes_by_track_index.items():
                pitch_range = S.pitch_range(tr_i=tr_i, measures=MIs)
                if pitch_range is not None:
                    is_drum = S.tracks[tr_i].is_drum
                    hi = pitch_range[1]
                    lo = pitch_range[0]
                    hi = min(127, hi)
                    lo = max(0, lo)
                    commands_at_end[tr_i] += enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT, is_drum=is_drum)
                    commands_at_end[tr_i] += enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT, is_drum=is_drum)

    return enc.encode_midisongbymeasure_with_masks(S=S,
                                                   note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE),
                                                   mask_locations=mask_locations,
                                                   measure_slice=measure_slice,
                                                   include_heads_for_empty_masked_measures=False,
                                                   track_measure_commands=track_measure_commands,
                                                   explicit_rhythmic_conditioning_locations=rhy_cond_locs,
                                                   rhythmic_conditioning_type=rhythmic_conditioning_type,
                                                   return_labels_too=return_labels_too,
                                                   extra_id_st=0,
                                                   extra_id_max=255,
                                                   commands_at_end=commands_at_end)


def is_good_for_val_test_infill_old(input_ids: "str", labels: "str", n_measures: "int"):
    raise NotImplemented("need a better test now that there can be N's in conditioning")
    n_measures_with_notes_in_example = 0  # masked and unmasked notes both count
    m_split = input_ids.split(';M')
    for s in m_split:
        if ';N:' in s or ';D:' in s or ';<extra_id' in s:
            n_measures_with_notes_in_example += 1

    # 4 and 8 measure examples require note onsets in all measures.
    # 16 measure examples require note onsets in all but 1 measure.
    # 32 measure examples require note onsets in all but 2 measures.
    if n_measures_with_notes_in_example >= n_measures - n_measures//16:
        return True

    return False


# test written
def aug_bpm(S: "ms.MidiSongByMeasure"):
    """in place operation"""
    amt = random.randint(-1, 1) * 5 * random.random() * .01
    for t in S.tempo_changes:
        t.val += amt * t.val


# test written
def aug_vel(S: "ms.MidiSongByMeasure"):
    """in place operation"""
    def fix_vel(v):
        v = int(v)
        if v < 1:
            v = 1
        if v > 127:
            v = 127
        return v

    amt = random.randint(-1, 1) * 5 * random.random() * .01
    for track in S.tracks:
        for tbm in track.tracks_by_measure:
            for n in tbm.note_ons:
                n.vel += amt * n.vel
                n.vel = fix_vel(n.vel)


# def aug_bpm_multiply_by_factor(S: "ms.MidiSongByMeasure", factor=2.0) -> "ms.MidiSongByMeasure":
#     """Returns a new MidiSongByMeasure that 'sounds' the same as S, but where all BPM's are multiplied by the given
#     factor and event lengths/positions are modified to compensate for that. Note that factors other than 0.5 and 2.0
#     give strange results.
#
#     For factor = 0.5, eighth notes become 16th notes (for example).
#     For factor = 2.0, eighth notes become quarter notes (for example).
#
#     For factor = 0.5, each new measure now contains two old measures. This may merge awkwardly.
#     For factor = 2.0, each new measure now contains half an old measure.
#     """
#     S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=False)
#     for t in S.tempo_changes:
#         t.val = t.val * factor
#     for evt_iterable in S.all_iterables_with_time_events():
#         for evt in evt_iterable:
#             if factor > 1:
#                 if hasattr(evt, 'end'):
#                     evt.end = int(evt.end * factor)
#                 evt.click = int(evt.click * factor)
#             else:
#                 evt.click = int(evt.click * factor)
#                 if hasattr(evt, 'end'):
#                     evt.end = int(evt.end * factor)
#     S = ms.MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
#     return S

def aug_4_4_to_1_4(S: "ms.MidiSongByMeasure") -> "ms.MidiSongByMeasure":
    """Returns new MidiSongByMeasure"""
    MEs = S.get_measure_endpoints(make_copy=True)
    MLs = S.get_measure_lengths()
    len_qn = S.cpq
    len_4_4 = 4 * len_qn
    new_MEs = []
    for ml, me in zip(MLs, MEs):
        new_MEs.append(me)
        if ml == len_4_4 and random.random() < P_4_4_to_1_4:
            for _ in range(3):
                new_MEs.append(new_MEs[-1] + len_qn)
    new_MEs.append(MEs[-1])  # there is one more measure endpoint than measure length from the original S

    # clean_up_time_signatures value doesn't matter, since we'll be ignoring the time signatures
    # in favor of the new measure endpoints we created anyway
    S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=False, clean_up_time_signatures=False)
    S = ms.MidiSongByMeasure.from_MidiSong(S, measure_endpoints=new_MEs, consume_calling_song=True)
    return S


def aug_4_4_to_2_4(S: "ms.MidiSongByMeasure") -> "ms.MidiSongByMeasure":
    """Returns new MidiSongByMeasure"""
    MEs = S.get_measure_endpoints(make_copy=True)
    MLs = S.get_measure_lengths()
    len_qn = S.cpq
    len_4_4 = 4 * len_qn
    new_MEs = []
    for ml, me in zip(MLs, MEs):
        new_MEs.append(me)
        if ml == len_4_4 and random.random() < P_4_4_to_2_4:
            for _ in range(1):
                new_MEs.append(new_MEs[-1] + 2 * len_qn)
    new_MEs.append(MEs[-1])  # there is one more measure endpoint than measure length from the original S

    # clean_up_time_signatures value doesn't matter, since we'll be ignoring the time signatures
    # in favor of the new measure endpoints we created anyway
    S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=False, clean_up_time_signatures=False)
    S = ms.MidiSongByMeasure.from_MidiSong(S, measure_endpoints=new_MEs, consume_calling_song=True)
    return S


def aug_4_4_to_8_4(S: "ms.MidiSongByMeasure") -> "ms.MidiSongByMeasure":
    MEs = S.get_measure_endpoints(make_copy=True)
    len_qn = S.cpq
    len_4_4 = 4 * len_qn
    new_MEs = []
    for i, me in enumerate(MEs):
        new_MEs.append(me)
        # if we have two adjacent measures of 4/4 and the probability threshold is met, then combine them to one
        # measure of 8/4
        if i > 1 and (me - new_MEs[-2]) == len_4_4 and (new_MEs[-2] - new_MEs[-3] == len_4_4):
            if random.random() < P_4_4_to_8_4:
                new_MEs.pop(-2)

    # clean_up_time_signatures value doesn't matter, since we'll be ignoring the time signatures
    # in favor of the new measure endpoints we created anyway
    S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=False, clean_up_time_signatures=False)
    S = ms.MidiSongByMeasure.from_MidiSong(S, measure_endpoints=new_MEs, consume_calling_song=True)
    return S


# test written
# TODO update this if we ever use more than 256 extra id's
def uncorrupt(input_ids: "list[int]", labels: "list[int]", tokenizer) -> "list[int]":
    """input_ids and labels lists of integers. Returns a list of integers."""
    corruption_markers = []
    extra_ids = [';<extra_id_{}>'.format(x) for x in range(256)]
    for marker in tokenizer.Encode(extra_ids):
        corruption_markers.append(marker[0])
    corruption_markers_set = set(corruption_markers)

    res = []
    len_labels = len(labels)
    for i, e in enumerate(input_ids):
        if e in corruption_markers_set:
            done = False
            try:
                i_labels = labels.index(e)
            except ValueError:
                done = True
            while not done:
                i_labels += 1
                if i_labels >= len_labels:
                    done = True

                if not done:
                    e_labels = labels[i_labels]
                    if e_labels in corruption_markers_set:
                        done = True

                if not done:
                    if e_labels != tokenizer.eos_id():
                        res.append(e_labels)
        else:
            res.append(e)
    return res


# test written
# TODO update this if we ever use more than 256 extra id's
def corrupt_pretrain(list_of_ints, tokenizer):
    """for corrupted span objective"""
    n_corruptions = math.floor(0.15*len(list_of_ints)/3)  # 0 or more corruptions
    n_corruptions = min(256, n_corruptions)  # at most 256 corruptions

    labels = []
    corrupted = []

    # first, get corruption indices
    sample_population = range(len(list_of_ints) - 2 * n_corruptions - 1 - n_corruptions)
    indices = sorted(random.sample(sample_population, n_corruptions))
    corruption_start_indices = []
    for i, x in enumerate(indices):
        corruption_start_indices.append(x + 1 + 3 * i)

    i = 0
    n_spans_done = 0
    done = False
    # iterate over L from left to right; randomize first corruption marker
    first_extra_id = random.randint(0, 256-n_corruptions)  # assumes tokenizer has extra_id's 0 thru 255
    while not done:
        if i in corruption_start_indices:
            corrupt_marker = tokenizer.encode(';<extra_id_{}>'.format(first_extra_id + n_spans_done))[0]
            labels.append(corrupt_marker)
            corrupted.append(corrupt_marker)
            labels.extend(list_of_ints[i:i + 3])
            n_spans_done += 1
            i += 3
        else:
            corrupted.append(list_of_ints[i])
            i += 1

        done = i >= len(list_of_ints)

    return corrupted, labels


# test written
def get_random_mask(S: "ms.MidiSongByMeasure",
                    measure_slice: "tuple[int, int]" = None,
                    pattern_type: int = 0,
                    extra_params: "dict" = None,
                    mask_only_track_measures_with_note_ons: bool = False,
                    ) -> "set[tuple[int, int]]":
    """returns a set of tuples of the form (track_index, measure_index)"""
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    if extra_params is None:
        extra_params = {}

    res = set()

    if pattern_type == 0:  # random measures and instruments.
        p = extra_params['mask_probability']

        for track_index in range(len(S.tracks)):
            for measure_index in range(*measure_slice):
                if random.random() < p:
                    res.add((track_index, measure_index))

    elif pattern_type == 1:  # random tracks, all measures
        if len(S.tracks) < 2:
            return res
        max_n_tracks_to_mask = extra_params.get('max_n_tracks_to_mask', len(S.tracks) - 1)
        n_tracks_to_mask = random.randint(1, max_n_tracks_to_mask)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for measure_index in range(*measure_slice):
            for track_index in tracks_to_mask:
                res.add((track_index, measure_index))

    elif pattern_type == 2:  # random measure, all tracks
        if 'masked_measure' in extra_params:
            measure_to_mask = extra_params['masked_measure']
        else:
            measure_to_mask = random.randint(measure_slice[0], measure_slice[1] - 1)

        for track_index in range(len(S.tracks)):
            res.add((track_index, measure_to_mask))

    elif pattern_type == 3:  # random measure, most tracks
        if len(S.tracks) // 2 > len(S.tracks) - 1:
            return res
        measure_to_mask = random.randint(measure_slice[0], measure_slice[1] - 1)
        n_tracks_to_mask = random.randint(len(S.tracks) // 2, len(S.tracks) - 1)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for track_index in tracks_to_mask:
            res.add((track_index, measure_to_mask))

    elif pattern_type == 4:  # 2, 3, or 4 consecutive measures chosen at random, all tracks
        m_slice_size = measure_slice[1] - measure_slice[0]
        if m_slice_size < 3:
            return res
        if 'masked_measures' in extra_params:
            masked_measures = extra_params['masked_measures']
        else:
            n_masked_measures = random.randint(2, min(m_slice_size, 4))
            first_masked_measure = random.randint(0, m_slice_size - n_masked_measures)
            masked_measures = [first_masked_measure + x for x in range(n_masked_measures)]

        for measure_index in masked_measures:
            for track_index in range(len(S.tracks)):
                res.add((track_index, measure_index))

    elif pattern_type == 5:  # 2, 3, or 4 consecutive measures chosen at random, most tracks
        m_slice_size = measure_slice[1] - measure_slice[0]

        if m_slice_size < 3:
            return res

        if len(S.tracks) // 2 > len(S.tracks)-1:
            return res

        if 'masked_measures' in extra_params:
            masked_measures = extra_params['masked_measures']
        else:
            n_masked_measures = random.randint(2, min(m_slice_size, 4))
            first_masked_measure = random.randint(0, m_slice_size - n_masked_measures)
            masked_measures = [first_masked_measure + x for x in range(n_masked_measures)]

        n_tracks_to_mask = random.randint(len(S.tracks) // 2, len(S.tracks) - 1)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for measure_index in masked_measures:
            for track_index in tracks_to_mask:
                res.add((track_index, measure_index))

    elif pattern_type == 6:  # random tracks; each track gets 1 or 2 random spans.
        if len(S.tracks) == 0:
            return res
        n_tracks_to_mask = random.randint(1, len(S.tracks))
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for track_index in tracks_to_mask:
            n_spans = random.randint(1, 2)
            if n_spans == 1:
                st_measure = random.randint(measure_slice[0], measure_slice[1] - 1)
                end_measure = random.randint(st_measure + 1, measure_slice[1])  # set up for range()
                # ignore the following; full masking is fine.
                # # make sure this is not a full masking of the instrument...
                # if st_measure == measure_slice[0] and end_measure == measure_slice[1]:
                #     # ...unless there's only one measure
                #     if end_measure - st_measure > 1:
                #         # then randomly increment the st measure or decrement the end measure
                #         if random.randint(0, 1):
                #             st_measure += 1
                #         else:
                #             end_measure -= 1

                for measure_index in range(st_measure, end_measure):
                    res.add((track_index, measure_index))

            elif n_spans == 2:
                if measure_slice[1] - measure_slice[0] < 3:
                    return res
                span_1_st = measure_slice[0]
                span_2_end = measure_slice[1]
                if random.randint(0, 1):
                    # then choose span 1 end first
                    span_1_end = random.randint(span_1_st + 1, span_2_end - 2)
                    span_2_st = random.randint(span_1_end + 1, span_2_end - 1)
                else:
                    # then choose span 2 st first
                    span_2_st = random.randint(span_1_st + 2, span_2_end - 1)
                    span_1_end = random.randint(span_1_st + 1, span_2_st - 1)

                for measure_index in range(span_1_st, span_1_end):
                    res.add((track_index, measure_index))
                for measure_index in range(span_2_st, span_2_end):
                    res.add((track_index, measure_index))

    if mask_only_track_measures_with_note_ons:
        real_res = set()
        for T in res:
            tr_i, measure_i = T
            if S.tracks[tr_i].tracks_by_measure[measure_i].note_ons:
                real_res.add(T)
        res = real_res

    return res


# test written
def get_trans_amt(epoch, i):
    trans_range = cs.AUG_TRANS_MAX - cs.AUG_TRANS_MIN + 1
    trans_amt = (epoch + i) % trans_range
    trans_amt += cs.AUG_TRANS_MIN
    return trans_amt


# helper function for build_val_test_finetune_data_infill
def _build_val_test_finetune_data_infill(T):
    i, k, v, tokenizer, epoch, mask_pattern_type, n_measures = T

    mask_pattern_type_str = str(mask_pattern_type)

    S = pre.midisongbymeasure_from_save_dict(v)

    if n_measures == 'all':
        n_measures = S.get_n_measures()

    if n_measures > S.get_n_measures():
        print('Notice: Could not generate example (not enough measures): Wanted {} measures; {}'.format(n_measures, k))
        return []

    # do not transpose val/test data for evaluating finetuned model
    # do, however, transpose instrument parts into acceptable ranges
    enc.transpose_into_acceptable_ranges_TT(S)
    S.sort_tracks_by_inst_and_avg_note_pitch()
    for t in S.tracks:
        t.sort()

    extra_id_max = 255

    n_tries = 25
    for try_ in range(n_tries):
        measure_st = random.randint(0, S.get_n_measures() - n_measures)
        measure_end = measure_st + n_measures
        measure_slice = (measure_st, measure_end)

        if mask_pattern_type_str == '2random':
            mask_pattern_int = 2
        elif mask_pattern_type_str == '2last':
            mask_pattern_int = 2
        elif mask_pattern_type_str == '0half':
            mask_pattern_int = 0
        elif mask_pattern_type_str == '0quarter':
            mask_pattern_int = 0
        elif mask_pattern_type_str == '1singleinst':
            mask_pattern_int = 1
        else:
            mask_pattern_int = int(mask_pattern_type)

        extra_params = None
        if mask_pattern_type_str == '2last':
            extra_params = {'masked_measure': measure_end - 1}
        elif mask_pattern_type_str == '4':
            extra_params = {'masked_measures': [measure_end - 2, measure_end - 1]}
        elif mask_pattern_type_str == "0half":
            extra_params = {'mask_probability': 0.5}
        elif mask_pattern_type_str == '0quarter':
            extra_params = {'mask_probability': 0.25}
        elif mask_pattern_type_str == '1':
            extra_params = {'max_n_tracks_to_mask': len(S.tracks)//2}
        elif mask_pattern_type_str == '1singleinst':
            extra_params = {'max_n_tracks_to_mask': 1}

        mask = get_random_mask(S, measure_slice=measure_slice, pattern_type=mask_pattern_int,
                               extra_params=extra_params, mask_only_track_measures_with_note_ons=True)

        unmasked_tr_measures = set()
        for tr_i, tr in enumerate(S.tracks):
            for m_i in range(*measure_slice):
                t = tr.tracks_by_measure[m_i]
                if t.note_ons and (tr_i, m_i) not in mask:
                    unmasked_tr_measures.add((tr_i, m_i))

        if is_good_for_val_test_infill(S, tokenizer, mask_pattern_type_str, mask, unmasked_tr_measures, measure_st, n_measures):
            # do the actual encoding and tokenization during validation/testing
            return [(k, {'processed_source': k,
                         'measure_slice': measure_slice,
                         'mask': list(mask),  # sets are not json serializable
                         'unmasked_tr_measures': list(unmasked_tr_measures),
                         'note_off_treatment': spm_type_to_note_off_treatment(cs.SPM_TYPE),
                         'transpose': 0
                         })]

    to_print = 'Notice: Could not generate example after {} tries {}'.format(n_tries, k)
    to_print += ' params: mask_pattern_type={}, n_measures={}'.format(mask_pattern_type_str, n_measures)
    to_print += ' last generated mask: {}'.format(mask)
    print(to_print)
    return []


def build_val_test_finetune_data_infill(tokenizer, epoch, pool, mode,
                                        mask_pattern_type=0,
                                        n_measures=5):
    """
    mode = 'val' or 'test'
    mask_pattern_type = 0 or 1 or "2random" or "2last" or 4 or 6
    n_measures = 5 or 9 or 17 or 33 or whatever or 'all'
    """
    t0 = time.time()

    P = pool
    to_write = []

    if mode == 'val':
        path = cs.PATH_TO_PROCESSED_VAL_MIDI
    elif mode == 'test':
        path = cs.PATH_TO_PROCESSED_TEST_MIDI
    else:
        raise ValueError('mode not recognized: {}'.format(mode))

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch, mask_pattern_type, n_measures) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            for i, res in enumerate(P.imap_unordered(_build_val_test_finetune_data_infill, items, chunksize=10)):
                to_write.extend(res)

                if (i + 1) % 1000 == 0:
                    print(i + 1, 'songs processed from this file so far')

    # to compensate for random return order from imap_unordered
    to_write.sort(key=lambda x: x[0])
    to_write = [x[1] for x in to_write]
    dict_to_write = {}
    for i, thing in enumerate(to_write):
        dict_to_write[i] = thing

    target_dir = os.path.join(cs.PATH_TO_TEMP_FILES, 'infill')
    os.makedirs(target_dir, exist_ok=True)

    if mode == 'val':
        with open(os.path.join(target_dir, 'finetune_validation_{}_{}.txt'.format(mask_pattern_type, n_measures)), 'w') as outfile:
            json.dump(dict_to_write, outfile)
    elif mode == 'test':
        with open(os.path.join(target_dir, 'finetune_test_{}_{}.txt'.format(mask_pattern_type, n_measures)), 'w') as outfile:
            json.dump(dict_to_write, outfile)
    to_print = 'finished building {} data ('.format(mode)
    to_print += 'mask_pattern_type={}'.format(mask_pattern_type)
    to_print += ', n_measures={})'.format(n_measures)
    to_print += ' in {} sec'.format(time.time() - t0)
    print(to_print)


# test written
def weighted_choose_one(dict_of_weights: dict):
    """dict of weights a dict with keys the objects you are trying to choose one from, and associated keys
    positive numbers"""
    total = sum(v for v in dict_of_weights.values())
    items = sorted(list(dict_of_weights.keys()))
    index_weights = [0]
    for k in items:
        if dict_of_weights[k] < 0:
            raise ValueError('all weights must be nonnegative')
        index_weights.append(index_weights[-1] + dict_of_weights[k])
    index = random.random() * total
    index = bisect.bisect_right(index_weights, index) - 1
    return items[index]


def random_permutation(L: list):
    return random.sample(L, len(L))


def _build_finetune_train_data(T):
    if cs.FINETUNE_TASK == 'infill':
        return _build_finetune_train_data_infill(T)
    else:
        raise NotImplemented(f'task = {cs.FINETUNE_TASK} not implemented')


def _update_commands_at_end(S: ms.MidiSongByMeasure,
                            commands_at_end: dict[int, str],
                            mask_dict_to_use: dict[int, set[int]],
                            include_commands_at_end: str,
                            measurement_str: str):
    for tr_i, MIs in mask_dict_to_use.items():
        if include_commands_at_end == 'all' or (include_commands_at_end == 'some' and random.random() < 0.5):
            val = enc.get_binned_measurement_value(S=S, tr_i=tr_i, measures=MIs,
                                                   measurement_str=measurement_str)
            if val is not None:
                commands_at_end[tr_i] += enc.instruction_str(val, measurement_str)


def _build_finetune_train_data_recursive_helper_infill(S: ms.MidiSongByMeasure,
                                                       source: str,
                                                       mask_pattern_type: int,
                                                       p_extend_one_more_measure: float,
                                                       p_truncate: float,
                                                       extra_id_max: int,
                                                       tokenizer,
                                                       is_first_example: bool,
                                                       p_first_example_starts_at_measure_0: float,
                                                       n_measures_to_get: int,
                                                       force_n_measures: int,
                                                       p_octave_shift_instructions_per_track_measure: float,
                                                       p_note_range_instr_per_track_measure: float,
                                                       p_explicit_rhythmic_conditioning_per_track_measure: float,
                                                       rhythmic_conditioning_type: str or None,
                                                       include_commands_at_end: str,
                                                       try_countdown: int):
    if try_countdown <= 0 or n_measures_to_get <= 0:
        return None

    if force_n_measures and S.get_n_measures() < n_measures_to_get:
        return None

    while random.random() < p_extend_one_more_measure:
        n_measures_to_get += 1
    n_measures_to_get = min(S.get_n_measures(), n_measures_to_get)  # can't get more measures than we have

    # decide the start measure for this example
    if is_first_example and random.random() < p_first_example_starts_at_measure_0:
        # first example starts at measure 0
        measure_st = 0
    else:
        # other examples start at random places
        measure_st = random.randint(0, S.get_n_measures() - n_measures_to_get)

    # artificially get fewer measures sometimes
    if random.random() < p_truncate:
        n_measures_to_drop = random.randint(0, n_measures_to_get - 1)
        n_measures_to_get -= n_measures_to_drop

    # get mask
    if mask_pattern_type == 0:
        extra_params = {'mask_probability': random.uniform(0.2, 0.8)}  # was just 0.5 in v1
    else:
        extra_params = None
    measure_slice = (measure_st, measure_st + n_measures_to_get)
    mask = get_random_mask(S, measure_slice=measure_slice,
                           pattern_type=mask_pattern_type,
                           extra_params=extra_params,
                           mask_only_track_measures_with_note_ons=True)
    # define extra_id_st
    if len(mask) > extra_id_max:
        extra_id_st = 0
    else:
        extra_id_st = random.randint(0, extra_id_max - len(mask) + 1)

    track_measure_commands = collections.defaultdict(str)

    # handle "not an octave shift of any track in this measure" commands
    for T in mask:
        tr_i, measure_i = T
        if random.random() < p_octave_shift_instructions_per_track_measure and not S.is_octave_collapse_of_some_track_in_this_measure(tr_i=tr_i, measure_i=measure_i):
            track_measure_commands[T] += enc.instruction_str(1, enc.MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE)

    # handle hi and lo note instructions per track measure
    for T in mask:
        tr_i, measure_i = T
        if random.random() < p_note_range_instr_per_track_measure:
            strict = random.random() < 0.5
            pitch_range = S.pitch_range(tr_i=tr_i, measures=[measure_i])
            if pitch_range is not None:
                lo, hi = pitch_range
                is_drum = S.tracks[tr_i].is_drum
                if strict:
                    instruction_lo = enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT, is_drum=is_drum)
                    instruction_hi = enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT, is_drum=is_drum)
                else:
                    lo -= random.randint(0, 7)
                    hi += random.randint(0, 7)
                    hi = min(127, hi)
                    lo = max(0, lo)
                    instruction_lo = enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE, is_drum=is_drum)
                    instruction_hi = enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE, is_drum=is_drum)
                track_measure_commands[T] += instruction_lo
                track_measure_commands[T] += instruction_hi

    # handle explicit rhythmic conditioning
    explicit_rhythmic_conditioning_locations = set()
    for T in mask:
        if random.random() < p_explicit_rhythmic_conditioning_per_track_measure:
            explicit_rhythmic_conditioning_locations.add(T)  # let string encoder handle these

    # handle density etc. user controls, one at a time
    commands_at_end = collections.defaultdict(str)
    if include_commands_at_end == 'none':
        pass
    elif include_commands_at_end in ('some', 'all'):
        # precompute some information
        masked_measure_indexes_by_track_index = collections.defaultdict(set)
        for T in mask:
            masked_measure_indexes_by_track_index[T[0]].add(T[1])

        masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index = collections.defaultdict(set)
        for T in explicit_rhythmic_conditioning_locations:
            masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index[T[0]].add(T[1])

        masked_measure_indexes_without_rhythmic_conditioning_by_track_index = collections.defaultdict(set)
        for tr_i, MIs in masked_measure_indexes_by_track_index.items():
            ERs = masked_measure_indexes_with_explicit_rhythmic_conditioning_by_track_index[tr_i]
            masked_measure_indexes_without_rhythmic_conditioning_by_track_index[tr_i] = MIs.difference(ERs)

        # there are three possibilities for rhythmic_conditioning_type:
        # - None
        # - 1d_flattening
        # - n_pitch_classes_and_n_notes

        # vert density:
        # - when rhythmic conditioning type is none, "always" include this
        #   (here and below, "always" means we compute it, and will include it if include_commands_at_end == 'all' or
        #   if include_commands_at_end == 'some' and _update_commands_at_end chooses to include it)
        # - when rhythmic conditioning type is 1-d flattening, always include this
        # - when rhythmic conditioning type is n_pitch_classes_and_n_notes, only include this for tracks where >= 1
        #   measure in that track doesn't include explicit rhythmic conditioning. The calculation of vert density
        #   applies only to measures without explicit rhythmic conditioning.
        if rhythmic_conditioning_type in (None, '1d_flattening'):
            mask_dict_to_use_for_vert_density = masked_measure_indexes_by_track_index
        elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
            mask_dict_to_use_for_vert_density = masked_measure_indexes_without_rhythmic_conditioning_by_track_index
        else:
            raise ValueError(f'rhythmic_conditioning_type={rhythmic_conditioning_type} not understood')
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=mask_dict_to_use_for_vert_density,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_VERT_NOTE_ONSET_DENSITY)

        # vert note onset n pitch classes on avg: when rhythmic conditioning type is n_pitch_classes_and_n_notes, only
        # include this for tracks where >= 1 measure in that track doesn't include explicit rhythmic conditioning.
        # In this case the measurement applies only to measures without explicit rhythmic conditioning.
        # For any other type of rhythmic conditioning, always include
        if rhythmic_conditioning_type in (None, '1d_flattening'):
            mask_dict_to_use_for_vert_n_pitch_classes_avg = masked_measure_indexes_by_track_index
        elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
            mask_dict_to_use_for_vert_n_pitch_classes_avg = masked_measure_indexes_without_rhythmic_conditioning_by_track_index
        else:
            raise ValueError(f'rhythmic_conditioning_type={rhythmic_conditioning_type} not understood')
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=mask_dict_to_use_for_vert_n_pitch_classes_avg,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG)

        # horiz density: only include this for tracks where >=1 measure in that track doesn't contain explicit
        # rhythmic conditioning. The calculation of horiz density applies only to track measures without explicit
        # rhythmic conditioning.
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY)

        # horiz note onset diversity percentage: same policy as horiz density
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE)

        # horiz note onset irregularity: same policy as horiz density
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_without_rhythmic_conditioning_by_track_index,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY)

        # pitch step and leap prob: always include
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_by_track_index,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_PITCH_STEP_PROB)
        _update_commands_at_end(S=S, commands_at_end=commands_at_end,
                                mask_dict_to_use=masked_measure_indexes_by_track_index,
                                include_commands_at_end=include_commands_at_end,
                                measurement_str=enc.MEASUREMENT_PITCH_LEAP_PROB)

        # highest note and lowest note
        for tr_i, MIs in masked_measure_indexes_by_track_index.items():
            pitch_range = S.pitch_range(tr_i=tr_i, measures=MIs)
            if pitch_range is not None:
                is_drum = S.tracks[tr_i].is_drum
                high_strict = random.random() < 0.5
                low_strict = random.random() < 0.5
                hi = pitch_range[1] if high_strict else pitch_range[1] + random.randint(0, 7)
                lo = pitch_range[0] if low_strict else pitch_range[0] - random.randint(0, 7)
                hi = min(127, hi)
                lo = max(0, lo)
                if include_commands_at_end == 'all' or (include_commands_at_end == 'some' and random.random() < 0.5):
                    if low_strict:
                        commands_at_end[tr_i] += enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT, is_drum=is_drum)
                    else:
                        commands_at_end[tr_i] += enc.instruction_str(lo, enc.ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE, is_drum=is_drum)
                if include_commands_at_end == 'all' or (include_commands_at_end == 'some' and random.random() < 0.5):
                    if high_strict:
                        commands_at_end[tr_i] += enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT, is_drum=is_drum)
                    else:
                        commands_at_end[tr_i] += enc.instruction_str(hi, enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE, is_drum=is_drum)

    else:
        raise ValueError(f'include_commands_at_end={include_commands_at_end} not recognized')

    continue_this_attempt = True
    input_ids_str, labels_str = '', ''
    mask = list(mask)
    if TRAIN_ON_RANDOM_PERMUTATIONS_OF_MASKS:
        mask = random_permutation(mask)
    else:
        mask.sort(key=lambda tup: (tup[1], tup[0]))

    try:
        input_ids_str, labels_str = enc.encode_midisongbymeasure_with_masks(
            S=S,
            note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE),
            mask_locations=mask,
            measure_slice=measure_slice,
            include_heads_for_empty_masked_measures=False,
            track_measure_commands=track_measure_commands,
            explicit_rhythmic_conditioning_locations=explicit_rhythmic_conditioning_locations,
            rhythmic_conditioning_type=rhythmic_conditioning_type,
            return_labels_too=True,
            extra_id_st=extra_id_st,
            extra_id_max=extra_id_max,
            commands_at_end=commands_at_end)
    except ValueError as err:
        print(f'Notice: ValueError in encode_midisongbymeasure_with_masks: {err}')
        print(f'len(mask_locations)={len(mask)}')
        print(f'source={source}')
        print(f'trying again; countdown index = {try_countdown}')
        continue_this_attempt = False

    input_ids, labels = [], []
    if continue_this_attempt:
        input_ids = tokenizer.Encode(input_ids_str)
        labels = tokenizer.encode(labels_str)
        if len(input_ids) > cs.MAX_LEN or len(labels) > cs.MAX_LEN - 1:
            continue_this_attempt = False

    this_res = {}
    if continue_this_attempt:
        if (';N:' in input_ids_str or ';D:' in input_ids_str) and labels:
            this_res = {'input_ids': input_ids, 'labels': labels + [tokenizer.eos_id()],
                        # measure_slice and mask are potentially inaccurate relative to the source S b/c of
                        # 1/4, 2/4 , 8/4 augmentation probabilities, so we do not record them
                        # 'measure_slice': measure_slice,
                        # 'mask': list(mask)
                        }
        else:
            continue_this_attempt = False

    if continue_this_attempt:
        if not passes_final_finetune_example_check(input_ids_str, labels_str, tokenizer):
            continue_this_attempt = False

    if continue_this_attempt:
        # print('created training data:')
        # print('input:', input_ids_str)
        # print('labels:', labels_str)
        return this_res

    else:
        if n_measures_to_get > 1 and not force_n_measures:
            input_ids = tokenizer.Encode(input_ids_str)
            labels = tokenizer.encode(labels_str)
            total_n_tokens = len(input_ids) + len(labels)
            n_tokens_per_measure_avg = total_n_tokens / n_measures_to_get

            if total_n_tokens > 2 * cs.MAX_LEN:
                n_measures_to_get = int((2 * cs.MAX_LEN) // n_tokens_per_measure_avg)
                n_measures_to_get += random.randint(-3, 3)
                n_measures_to_get = 1 if n_measures_to_get < 1 else n_measures_to_get

            else:
                n_measures_to_get -= 1

        return _build_finetune_train_data_recursive_helper_infill(S=S,
                                                                  source=source,
                                                                  mask_pattern_type=mask_pattern_type,
                                                                  p_extend_one_more_measure=p_extend_one_more_measure,
                                                                  p_truncate=p_truncate,
                                                                  extra_id_max=extra_id_max,
                                                                  tokenizer=tokenizer,
                                                                  is_first_example=is_first_example,
                                                                  p_first_example_starts_at_measure_0=p_first_example_starts_at_measure_0,
                                                                  n_measures_to_get=n_measures_to_get,
                                                                  force_n_measures=force_n_measures,
                                                                  p_octave_shift_instructions_per_track_measure=p_octave_shift_instructions_per_track_measure,
                                                                  p_note_range_instr_per_track_measure=p_note_range_instr_per_track_measure,
                                                                  p_explicit_rhythmic_conditioning_per_track_measure=p_explicit_rhythmic_conditioning_per_track_measure,
                                                                  rhythmic_conditioning_type=rhythmic_conditioning_type,
                                                                  include_commands_at_end=include_commands_at_end,
                                                                  try_countdown=try_countdown - 1)


def _build_finetune_train_data_infill(T):
    i_, k_, v_, tokenizer, epoch, force_n_measures = T
    # setup
    S_orig = pre.midisongbymeasure_from_save_dict(v_)

    n_examples_to_get = 1 + S_orig.get_n_measures() // 8  # get at least 1 example per song; this denominator was 16 for v1 of the paper
    n_examples_to_get = min(16, n_examples_to_get)  # get at most 16 examples per song
    n_examples_to_get *= get_finetune_examples_multiplier(S=S_orig, p=k_)
    n_examples_to_get = round(n_examples_to_get)

    sampling_prob = finetune_sampling_prob(S=S_orig, p=k_)
    for x in range(n_examples_to_get):
        if random.random() > sampling_prob:
            n_examples_to_get -= 1

    n_tries = n_examples_to_get * 2

    # probabilities
    p_first_example_starts_at_measure_0 = 0.8
    p_drop_tracks = 0.0  # was 0.0 for ISMIR 2023 paper. A subsequent experiment reveals that 0.0 performs better than 0.8 for both the Lakh and the open datasets, both for full-slice infilling and dropped-track-slice infilling
    p_extend_one_more_measure = 0.7  # must be < 1
    if force_n_measures:
        p_extend_one_more_measure = 0.0
    p_truncate = 0.15
    if force_n_measures:
        p_truncate = 0.0

    p_include_octave_shift_instructions = 0.8  # exclude "not an octave shift of any other track in this measure" instructions completely from an example with 1 - this prob
    p_include_octave_shift_instruction_per_track_measure = 0.7  # when they are included, include them with this probability per track measure (independently sampled).

    p_include_note_range_instructions_in_track_measures = 0.5  # include explicit low/high notes on a per-track-measure basis for masked track measures
    p_include_note_range_instruction_per_track_measure = 0.9  # when they are included in an example, include them per track measure with this probability

    p_include_explicit_rhythmic_conditioning = 0.5  # exclude rhythmic conditioning completely from an example with 1 - this prob
    p_explicit_rhythmic_conditioning_per_track_measure = 0.85  # only applies for examples with explicit rhythmic conditioning
    rhythmic_conditioning_weights = {'1d_flattening': 1, 'n_pitch_classes_and_n_notes': 1}  # there are two types of rhythmic conditioning. When including explicit rhythmic conditioning, pick between them according to these weights

    # mask pattern weights
    pattern_weights = {0: 4, 1: 6, 2: 1, 3: 1, 4: 1, 5: 1, 6: 4}

    # always "none" was the training setup for the ISMIR 2023 paper. Commands at end include horizontal and vertical
    # note density, and step and leap propensity
    include_commands_at_end_weights = {'some': 2, 'all': 2, 'none': 1}

    extra_id_max = 255
    res = []

    if len(S_orig.tracks) == 0 or n_examples_to_get <= 0:
        return res

    done = False
    n_tries_so_far = 0
    while not done:
        S = copy.copy(S_orig)

        if random.random() < p_drop_tracks and len(S.tracks) > 1:
            # then drop 0 or more tracks. Do not drop all tracks.
            tracks_to_drop = random.sample(range(len(S.tracks)), random.randint(0, len(S.tracks) - 1))
            tracks_to_drop.sort()
            for i, pop_val in enumerate(tracks_to_drop):
                S.tracks.pop(pop_val - i)

        transpose_amt = get_trans_amt(epoch=epoch, i=i_ + len(res))
        S.transpose(amt=transpose_amt)
        enc.transpose_into_acceptable_ranges_TT(S)
        aug_bpm(S)  # each example we generate from this song gets a different BPM
        aug_vel(S)  # each example we generate from this song gets a different velocity augmentation
        S.sort_tracks_by_inst_and_avg_note_pitch()
        for t in S.tracks:
            t.sort()

        # augment 4/4 to other time signatures according to the probabilities at the top of this file
        S = aug_4_4_to_8_4(S)
        S = aug_4_4_to_2_4(S)
        S = aug_4_4_to_1_4(S)

        if len(S.tracks) == 1:
            pattern_type = weighted_choose_one({0: pattern_weights[0], 4: pattern_weights[4], 6: pattern_weights[6]})
        else:
            pattern_type = weighted_choose_one(pattern_weights)

        include_octave_shift_instructions = random.random() < p_include_octave_shift_instructions
        if include_octave_shift_instructions:
            p_octave_shift_instr = p_include_octave_shift_instruction_per_track_measure
        else:
            p_octave_shift_instr = 0.0

        include_note_range_instructions_in_track_measures = random.random() < p_include_note_range_instructions_in_track_measures
        if include_note_range_instructions_in_track_measures:
            p_note_range_instr_per_track_measure = p_include_note_range_instruction_per_track_measure
        else:
            p_note_range_instr_per_track_measure = 0.0

        include_commands_at_end = weighted_choose_one(include_commands_at_end_weights)

        n_measures_to_get = force_n_measures if force_n_measures else 32
        # the following two lines were not there for v1 of the first paper
        if not force_n_measures:
            n_measures_to_get = weighted_choose_one({4: 2, 8: 4, 12: 4, 16: 2, 20: 1})

        if random.random() < p_include_explicit_rhythmic_conditioning:  # then we will include explicit rhythmic conditions
            rhythmic_conditioning_type = weighted_choose_one(rhythmic_conditioning_weights)
            p_rhythm_conditioning = p_explicit_rhythmic_conditioning_per_track_measure
        else:
            rhythmic_conditioning_type = None
            p_rhythm_conditioning = 0.0

        this_res = _build_finetune_train_data_recursive_helper_infill(S=S,
                                                                      source=k_,
                                                                      mask_pattern_type=pattern_type,
                                                                      p_extend_one_more_measure=p_extend_one_more_measure,
                                                                      p_truncate=p_truncate,
                                                                      extra_id_max=extra_id_max,
                                                                      tokenizer=tokenizer,
                                                                      is_first_example=len(res) == 0,
                                                                      p_first_example_starts_at_measure_0=p_first_example_starts_at_measure_0,
                                                                      n_measures_to_get=n_measures_to_get,
                                                                      force_n_measures=force_n_measures,
                                                                      p_octave_shift_instructions_per_track_measure=p_octave_shift_instr,
                                                                      p_note_range_instr_per_track_measure=p_note_range_instr_per_track_measure,
                                                                      p_explicit_rhythmic_conditioning_per_track_measure=p_rhythm_conditioning,
                                                                      rhythmic_conditioning_type=rhythmic_conditioning_type,
                                                                      include_commands_at_end=include_commands_at_end,
                                                                      try_countdown=10)

        if this_res is not None:
            # this_res['measure coverage'] = (tokenizer.decode(this_res['input_ids']).count(';M'), S_orig.get_n_measures())
            this_res['track coverage'] = (len(S.tracks), len(S_orig.tracks))
            this_res['pattern type'] = pattern_type
            this_res['source'] = k_
            res.append((k_, this_res))
        else:
            pass

        n_tries_so_far += 1
        done = (len(res) == n_examples_to_get) or (n_tries_so_far >= n_tries)

    # print('n examples found', len(res))
    return res


# TODO maybe - see scratchpad 54 for current progress
# def _build_finetune_train_data_arrange(T):
#     i_, k_, v_, tokenizer, epoch, force_n_measures = T
#     # setup
#     S = pre.midisongbymeasure_from_save_dict(v_)
#
#     p_truncate = 0.5
#
#     s = enc.encode_midisongbymeasure(S, note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE))
#     s_measures = s.split(';M:')
#     s_measures = [';M:' + x for x in s_measures]  # put the ';M:' back at the beginning of each measure
#     s_measure_lens = [len(tokenizer.encode(x)) for x in s_measures]


def build_finetune_train_data(tokenizer, epoch, pool, path, force_n_measures: bool or int = False):
    t0 = time.time()
    print('Building finetune train data for task = {}, epoch = {}'.format(cs.FINETUNE_TASK, epoch))

    P = pool

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch, force_n_measures) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            to_write = []
            for i, res in enumerate(P.imap_unordered(_build_finetune_train_data, items, chunksize=1)):
                to_write.extend(res)

                if (i + 1) % 100 == 0:
                    print(i + 1, 'songs processed from this file so far')

            print('writing finetune train data for task = {}, epoch = {}'.format(cs.FINETUNE_TASK, epoch))
            # to compensate for random return order from imap_unordered
            to_write.sort(key=lambda x: x[0])
            to_write = [x[1] for x in to_write]

            target_dir = os.path.join(cs.PATH_TO_TEMP_FILES, cs.FINETUNE_TASK)
            os.makedirs(target_dir, exist_ok=True)

            with open(os.path.join(target_dir, f'finetune_epoch_{epoch}_{fname}'), 'wb') as outfile:
                pickle.dump(to_write, outfile)

    print('finished building finetune train data for epoch {} in {} sec'.format(epoch, time.time() - t0))


# helper function for build_pretrain_data
# test written
def _build_pretrain_data(T):
    i, k, v, tokenizer, epoch, target_len = T
    # random.seed(epoch)
    S = pre.midisongbymeasure_from_save_dict(v)
    S.transpose(amt=get_trans_amt(epoch=epoch, i=i))
    enc.transpose_into_acceptable_ranges_TT(S)
    aug_bpm(S)
    aug_vel(S)
    S.sort_tracks_by_inst_and_avg_note_pitch()
    for t in S.tracks:
        t.sort()

    S = aug_4_4_to_8_4(S)
    S = aug_4_4_to_2_4(S)
    S = aug_4_4_to_1_4(S)

    s = enc.encode_midisongbymeasure(S, note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE))
    ints = tokenizer.Encode(s)
    i = 0
    done = False
    res = []
    while not done:
        sl = ints[i * target_len: (i + 1) * target_len]
        if sl:
            corrupted = corrupt_pretrain(list_of_ints=sl, tokenizer=tokenizer)
            if corrupted[0] and corrupted[1]:
                this_res = {'input_ids': corrupted[0], 'labels': corrupted[1] + [tokenizer.eos_id()]}
                res.append((k, this_res))
            i += 1
        else:
            done = True
    return res


def build_pretrain_data(tokenizer, epoch, pool, mode):
    t0 = time.time()

    if mode in ('train', 'val_short'):
        max_len = 512 if epoch < cs.N_EPOCHS_SHORT else cs.MAX_LEN
    elif mode == 'val_long':
        max_len = cs.MAX_LEN
    else:
        raise ValueError('mode {} not recognized'.format(mode))

    addl = ' epoch = {}'.format(epoch) if mode == 'train' else ''
    print('Building pretrain data for mode = {}{}'.format(mode, addl))

    target_len = math.floor(max_len * 1 / (.85 + .15 / 3))
    n_corruptions_max = math.floor(0.15 * target_len / 3)  # 0 or more corruptions
    n_corruptions_max = min(256, n_corruptions_max)  # at most 256 corruptions
    while target_len - 2 * n_corruptions_max > max_len:
        target_len -= 1
        n_corruptions_max = math.floor(0.15 * target_len / 3)  # 0 or more corruptions
        n_corruptions_max = min(256, n_corruptions_max)  # at most 256 corruptions

    P = pool
    to_write = []

    if mode == 'train':
        path = cs.PATH_TO_PROCESSED_TRAIN_MIDI
    elif mode in ('val_short', 'val_long'):
        path = cs.PATH_TO_PROCESSED_VAL_MIDI
    else:
        raise ValueError('mode = {} not recognized'.format(mode))

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch, target_len) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            for i, res in enumerate(P.imap_unordered(_build_pretrain_data, items, chunksize=10)):
                to_write.extend(res)

                if (i + 1) % 1000 == 0:
                    print(i + 1, 'songs processed from this file so far')

            if mode == 'train':
                print('writing pretrain data for epoch {}'.format(epoch))
                # to compensate for random return order from imap_unordered
                to_write.sort(key=lambda x: x[0])
                to_write = [x[1] for x in to_write]
                with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_epoch_{}_{}'.format(epoch, fname)), 'wb') as outfile:
                    pickle.dump(to_write, outfile)
                to_write = []

    # Training data is already written above in chunks. Here we handle writing validation data to a single file.
    if mode == 'val_short':
        print('writing pretrain validation data (short inputs)')
    elif mode == 'val_long':
        print('writing pretrain validation data (long inputs)')

    # to compensate for random return order from imap_unordered
    to_write.sort(key=lambda x: x[0])
    to_write = [x[1] for x in to_write]

    if mode == 'val_long':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_long.txt'), 'wb') as outfile:
            pickle.dump(to_write, outfile)
        print('finished building pretrain validation data in {} sec'.format(time.time() - t0))
    elif mode == 'val_short':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_short.txt'), 'wb') as outfile:
            pickle.dump(to_write, outfile)
        print('finished building pretrain validation data in {} sec'.format(time.time() - t0))
    elif mode == 'train':
        print('finished building pretrain training data for epoch {} in {} sec'.format(epoch, time.time() - t0))


class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, epoch=None, mode='train'):
        """mode = 'train' or 'val_short' or 'val_long'"""
        self.data = []
        # load ALL training data into memory
        t0 = time.time()
        if mode == 'train':
            for folder, _, fnames in os.walk(cs.PATH_TO_TEMP_FILES):
                for fname in fnames:
                    if fname.find('pretrain_epoch_{}_'.format(epoch)) == 0:
                        with open(os.path.join(folder, fname), 'rb') as infile:
                            print('Loading files for PreTrainDataset mode={} epoch={}...'.format(mode, epoch))
                            d = pickle.load(infile)
                            self.data.extend(d)
        elif mode == 'val_short':
            with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_short.txt'), 'rb') as infile:
                print('Loading files for PreTrainDataset mode={}...'.format(mode))
                d = pickle.load(infile)
                self.data.extend(d)
        elif mode == 'val_long':
            with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_long.txt'), 'rb') as infile:
                print('Loading files for PreTrainDataset mode={}...'.format(mode))
                d = pickle.load(infile)
                self.data.extend(d)

        if not self.data:
            if mode == 'train':
                err = 'pretrain epoch {}'.format(epoch)
            elif mode == 'val_short':
                err = 'validation (short sequences)'
            elif mode == 'val_long':
                err = 'validation (long sequences)'
            else:
                raise ValueError('mode {} not recognized'.format(mode))
            raise RuntimeError('No data for {} found. Did you run build_pretrain_data.py?'.format(err))

        n_tokens = 0
        for d in self.data:
            n_tokens += len(d['input_ids']) + len(d['labels'])
        print('PreTrainDataset containing {} examples ({} tokens; {} tokens/example) loaded in {} sec'.format(
            len(self.data), n_tokens, round(n_tokens/len(self.data), 1), time.time() - t0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class FineTuneTrainDatasetSubset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class FineTuneTrainDataset:
    def __init__(self, epoch, size_cutoffs: tuple[int, int] = (512, 1024),
                 max_len_override=None, slice_in_epoch=0, n_slices_in_epoch=1, remove_extraneous_keys=True):
        """max_len_override just for development purposes"""
        self.data = []

        # load ALL training data into memory
        t0 = time.time()
        where = os.path.join(cs.PATH_TO_TEMP_FILES, cs.FINETUNE_TASK)
        for folder, _, fnames in os.walk(where):
            for fname in fnames:
                if fname.find('finetune_epoch_{}_'.format(epoch)) == 0:
                    print('Loading files from {} for FineTuneTrainDataset'.format(os.path.join(folder, fname)))
                    with open(os.path.join(folder, fname), 'rb') as infile:
                        d = pickle.load(infile)
                        self.data.extend(d)

        if not self.data:
            raise ValueError('No data for finetune epoch {} (task = {}) found. Did you run build_finetune_train_data.py?'.format(epoch, cs.FINETUNE_TASK))

        # remove extraneous keys from finetune examples
        if remove_extraneous_keys:
            for d in self.data:
                keys_to_del = []
                for k in d:
                    if k not in ('input_ids', 'labels'):
                        keys_to_del.append(k)
                for k in keys_to_del:
                    del d[k]

        # handle max_len_override
        if max_len_override is not None:
            new_data = []
            for d in self.data:
                if len(d['input_ids']) < max_len_override and len(d['labels']) < max_len_override:
                    new_data.append(d)
            self.data = new_data

        # handle slice within epoch
        new_data = []
        for i, d in enumerate(self.data):
            if i % n_slices_in_epoch == slice_in_epoch:
                new_data.append(d)
        self.data = new_data

        # partition into three subsets based on size
        new_data = [[], [], []]
        for ex in self.data:
            if len(ex['input_ids']) <= size_cutoffs[0] and len(ex['labels']) <= size_cutoffs[0]:
                new_data[0].append(ex)
            elif len(ex['input_ids']) <= size_cutoffs[1] and len(ex['labels']) <= size_cutoffs[1]:
                new_data[1].append(ex)
            else:
                new_data[2].append(ex)
        self.data = [FineTuneTrainDatasetSubset(d) for d in new_data]

        n_examples = sum(len(d) for d in self.data)
        print(f'FineTuneTrainDataset containing {n_examples} examples loaded:')
        str_description = {0: 'Small', 1: 'Medium', 2: 'Large'}
        for i, L in enumerate(self.data):
            if L:
                n_tokens = 0
                max_n_tokens = 0
                for d in L:
                    n_tokens += len(d['input_ids']) + len(d['labels'])
                    max_n_tokens = max(max_n_tokens, len(d['input_ids']) + len(d['labels']))
                print(f'{str_description[i]} example subset contains {len(L)} examples; {n_tokens} tokens; '
                      f'{round(n_tokens/len(L), 1)} tokens/example; max number of tokens = {max_n_tokens}')

        print(f'FineTuneTrainDataset loaded in {time.time() - t0} sec')
        print(f'FineTuneTrainDataset is for task = {cs.FINETUNE_TASK}')

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def random_subset_of_size(self, size):
        raise NotImplemented
        # what follows is old code from before we did data subsetting based on size
        if size >= len(self):
            return self
        indexes = set()
        while len(indexes) < size:
            indexes.add(random.randint(0, len(self) - 1))
        indexes = sorted(list(indexes))
        res = torch.utils.data.Subset(self, indexes)
        return res


class FineTuneValTestDatasetInfill(torch.utils.data.Dataset):
    def __init__(self, mode, mask_pattern_type, n_measures):
        """
        mode = 'val' or 'test'

        mask_pattern_type in ("0half", "0quarter", 1, "1singleinst", "2random", "2last", 4, 6)

        n_measures either 'all' or an int
        """
        self.data = []

        t0 = time.time()

        if mode == 'val':
            s = 'finetune_validation_'
        elif mode == 'test':
            s = 'finetune_test_'
        else:
            raise ValueError('mode {} not recognized'.format(mode))

        s += '{}_'.format(mask_pattern_type)
        s += '{}.txt'.format(n_measures)

        path = os.path.join(cs.PATH_TO_TEMP_FILES, 'infill', s)
        if not os.path.exists(path):
            raise ValueError('No data file named {} found. Did you run build_val_and_test_finetune_data_infill.py?'.format(s))

        with open(path) as infile:
            d = json.load(infile)
        for k in sorted([int(x) for x in d]):
            this_example = d[str(k)]
            self.data.append(this_example)

        print('FineTuneValTestDatasetInfill containing {} examples loaded in {} sec'.format(len(self.data), time.time()-t0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def batch_padder(batch, tokenizer, max_padding=None, add_attention_mask=True):
    """uses tokenizer.pad_id() to pad on right.
    max_padding = an integer, if you ALWAYS want to pad to that amount. Otherwise, pads to max len in the batch.
    """
    keys_to_pad = ('input_ids', 'labels')

    res = collections.defaultdict(list)
    pad_id = tokenizer.pad_id()

    if max_padding is not None:
        max_input_len_dict = {k: max_padding for k in keys_to_pad}
    else:
        max_input_len_dict = {}
        for k in keys_to_pad:
            max_input_len = 0
            for b in batch:
                max_input_len = max(len(b[k]), max_input_len)
            max_input_len_dict[k] = max_input_len

    for b in batch:
        for k, v in b.items():
            if k == 'labels':
                to_pad = -100
            else:
                to_pad = pad_id

            res[k].append(v + [to_pad] * (max_input_len_dict[k] - len(v)))

        if add_attention_mask:
            k = 'input_ids'
            res['attention_mask'].append(
                [1] * len(b[k]) + [0] * (max_input_len_dict[k] - len(b[k]))
            )

    # convert to tensors:
    for k in res:
        res[k] = torch.tensor(res[k], dtype=torch.long, requires_grad=False)
    return dict(res)
