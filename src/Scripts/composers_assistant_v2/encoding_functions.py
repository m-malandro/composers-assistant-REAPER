import typing

import constants as cs
import midisong as ms
import collections
import bisect
import statistics as stat
import copy

# this file contains functions for encoding midisongbymeasure objects to strings.
# to encode such strings to sequences of integers, use nn_training_functions.get_tokenizer()

# learned from train data; transfers well to val data
BPM_SLICER = [59.00000885, 74.9, 89.999955, 105.00157502, 119.9, 138.45653273, 165.000165]

# learned from train data; transfers well to val data, except lowest dynamic level is underrepresented in val data
DYNAMICS_SLICER = [64.4, 76.66666667, 81.9, 89.36666667, 95.9, 100.5, 109.9]

# based on DYNAMICS_SLICER
DYNAMICS_DEFAULTS = [58, 70, 79, 85, 93, 98, 105, 115]


ENCODING_INSTRUCTION_INSTRUCTIONS_AT_END_SEP = "instructions_at_end_sep"  # 1 instruction
MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY = "horiz_note_onset_density"  # 6 instructions
MEASUREMENT_VERT_NOTE_ONSET_DENSITY = "vert_note_onset_density"  # 5 instructions
MEASUREMENT_PITCH_STEP_PROB = "pitch_step_prob"  # 7 instructions
MEASUREMENT_PITCH_LEAP_PROB = "pitch_leap_prob"  # 7 instructions
MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG = "vert_note_onset_n_pitch_classes_on_avg"  # 5 instructions
MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY = "horiz_note_onset_irregularity"  # 4 instructions
# MEASUREMENT_HORIZ_NOTE_ONSET_BY_MEASURE_STD = "horiz_note_onset_by_measure_std"
MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE = "horiz_note_onset_density_diversity_percentage"  # 4 instructions
MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE = "is_not_octave_same"  # 1 instruction
ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM = "replace_keeping_rhythm"  # 1 instruction
ENCODING_INSTRUCTION_RHYTHM_PLACEHOLDER = "rhythm_placeholder"  # 1 instruction
ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM_AND_N_NOTES_AND_N_PITCH_CLASSES = "replace_keeping_rhythm_and_n_notes_and_n_pitch_classes"  # 1 instruction
ENCODING_INSTRUCTION_DISTINCT_PITCH_CLASS_MARKER = "distinct_pitch_class_marker"  # 1 instruction
ENCODING_INSTRUCTION_EXTRA_NOTE_ONSET_MARKER = "extra_note_onset_marker"  # 1 instruction
ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT = "highest_note_strict"  # 1 instruction
ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT = "lowest_note_strict"  # 1 instruction
ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE = "highest_note_loose"  # 1 instruction
ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE = "lowest_note_loose"  # 1 instruction


def _build_measurement_and_encoding_instruction_to_instruction_dict() -> dict[tuple[str, int], int]:
    d = {}
    i = 0

    d[(ENCODING_INSTRUCTION_INSTRUCTIONS_AT_END_SEP, 0)] = i
    i += 1

    for x in range(len(cs.HORIZ_NOTE_ONSET_DENSITY_SLICES) + 1):
        d[(MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY, x)] = i
        i += 1
    for x in range(len(cs.VERT_NOTE_ONSET_DENSITY_SLICES) + 1):
        d[(MEASUREMENT_VERT_NOTE_ONSET_DENSITY, x)] = i
        i += 1
    for x in range(len(cs.PITCH_HIST_STEP_SLICES) + 1):
        d[(MEASUREMENT_PITCH_STEP_PROB, x)] = i
        i += 1
    for x in range(len(cs.PITCH_HIST_LEAP_SLICES) + 1):
        d[(MEASUREMENT_PITCH_LEAP_PROB, x)] = i
        i += 1
    for x in range(len(cs.VERT_NOTE_ONSET_N_PITCH_CLASSES_SLICES) + 1):
        d[(MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG, x)] = i
        i += 1
    for x in range(len(cs.HORIZ_NOTE_ONSET_IRREGULARITY_SLICES) + 1):
        d[(MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY, x)] = i
        i += 1
    # for x in range(len(cs.HORIZ_NOTE_ONSET_DENSITY_STD_SLICES) + 1):
    #     d[(MEASUREMENT_HORIZ_NOTE_ONSET_BY_MEASURE_STD, x)] = i
    #     i += 1
    for x in range(len(cs.HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE_SLICES) + 1):
        d[(MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE, x)] = i
        i += 1

    d[(MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE, 1)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_RHYTHM_PLACEHOLDER, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM_AND_N_NOTES_AND_N_PITCH_CLASSES, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_DISTINCT_PITCH_CLASS_MARKER, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_EXTRA_NOTE_ONSET_MARKER, 0)] = i
    i += 1

    d[(ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE, 0)] = i
    i += 1
    d[(ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE, 0)] = i
    i += 1

    return d


# encoding instructions are also considered to be "measurements"
_MEASUREMENT_TO_INSTRUCTION_DICT = _build_measurement_and_encoding_instruction_to_instruction_dict()
_INSTRUCTION_TO_MEASUREMENT_DICT = {v: k for k, v in _MEASUREMENT_TO_INSTRUCTION_DICT.items()}


def instruction_str(value: int,
                    measurement_str_or_encoding_instruction_str: str,
                    is_drum: bool = False) -> str:
    """Aside from highest/lowest note instructions, all *encoding* instruction strings have value 0."""
    if measurement_str_or_encoding_instruction_str in (ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT,
                                                       ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT,
                                                       ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE,
                                                       ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE,):
        i = _MEASUREMENT_TO_INSTRUCTION_DICT[(measurement_str_or_encoding_instruction_str, 0)]
        note_character = 'D' if is_drum else 'N'
        return f';<instruction_{i}>' + f';{note_character}:{value}'
    else:
        i = _MEASUREMENT_TO_INSTRUCTION_DICT[(measurement_str_or_encoding_instruction_str, value)]
        return f';<instruction_{i}>'


def get_binned_measurement_value(S: ms.MidiSongByMeasure, tr_i: int, measures: typing.Iterable[int],
                                 measurement_str: str = MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY) -> int or None:
    eps = 0.0001
    if measurement_str == MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY:
        raw_val = S.horiz_note_onset_density(tr_i=tr_i, measures=measures)
        L = cs.HORIZ_NOTE_ONSET_DENSITY_SLICES
    elif measurement_str == MEASUREMENT_VERT_NOTE_ONSET_DENSITY:
        raw_val = S.vert_note_onset_density(tr_i=tr_i, measures=measures)
        L = cs.VERT_NOTE_ONSET_DENSITY_SLICES
    elif measurement_str == MEASUREMENT_PITCH_STEP_PROB:
        raw_val = S.consolidated_pitch_interval_hist(tr_i=tr_i, measures=measures)['step']
        L = cs.PITCH_HIST_STEP_SLICES
    elif measurement_str == MEASUREMENT_PITCH_LEAP_PROB:
        raw_val = S.consolidated_pitch_interval_hist(tr_i=tr_i, measures=measures)['leap']
        L = cs.PITCH_HIST_LEAP_SLICES
    elif measurement_str == MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG:
        raw_val = S.vert_note_onset_n_pitch_classes_avg(tr_i=tr_i, measures=measures)
        L = cs.VERT_NOTE_ONSET_N_PITCH_CLASSES_SLICES
    elif measurement_str == MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY:
        raw_val = S.horiz_note_onset_irregularity(tr_i=tr_i, measures=measures)
        L = cs.HORIZ_NOTE_ONSET_IRREGULARITY_SLICES
    # elif measurement_str == MEASUREMENT_HORIZ_NOTE_ONSET_BY_MEASURE_STD:
    #     raw_val = S.horiz_note_onset_by_measure_stdev(tr_i=tr_i, measures=measures)
    #     L = cs.HORIZ_NOTE_ONSET_DENSITY_STD_SLICES
    elif measurement_str == MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE:
        raw_val = horizontal_note_onset_density_diversity_percentage(S=S, tr_i=tr_i, measures=measures)
        L = cs.HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE_SLICES
    else:
        raise ValueError(f"measurement_str={measurement_str} not understood")

    if raw_val is None:
        return None

    if measurement_str in (MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY,):
        raw_val += eps
    elif measurement_str in (MEASUREMENT_VERT_NOTE_ONSET_DENSITY, MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG):
        raw_val -= eps

    return bisect.bisect(L, raw_val)


def horizontal_note_onset_density_diversity_percentage(S: ms.MidiSongByMeasure, tr_i: int,
                                                       measures: typing.Iterable[int]) -> float or None:
    counter = collections.Counter()
    for m_i in measures:
        v = get_binned_measurement_value(S=S, tr_i=tr_i, measures=[m_i],
                                         measurement_str=MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY)
        if v is not None:
            counter[v] += 1

    counts = [val for val in counter.values()]

    if sum(counts) <= 1:  # need 2+ measures for this to make sense
        return None
    else:
        most_frequent = max(counts)
        denominator = sum(counts)
        most_frequent_index = counts.index(most_frequent)
        counts.pop(most_frequent_index)  # remove one instance of the most frequent count
        numerator = denominator - most_frequent
        return numerator / denominator


def get_bpm_level(b: float) -> int:
    """b a bmp float value. Returns an integer in [0,...,7]"""
    return bisect.bisect_right(BPM_SLICER, b)


def get_loudness_level(x: float) -> int:
    """x an average velocity level for a measure. Returns an integer in [0,...,7]"""
    return bisect.bisect_right(DYNAMICS_SLICER, x)


def _avg_vel_of_tr(tr):
    res = [n.vel for n in tr.note_ons]
    if res:
        return stat.mean(res)
    else:
        return None


def _avg_vel_of_measure(m):
    """empty measures are considered to have 0.0 avg vel"""
    res = [_avg_vel_of_tr(tr) for tr in m]
    res = [x for x in res if x is not None]
    if res:
        return stat.mean(res)
    else:
        return 0.0


def transpose_into_acceptable_ranges_TT(S: ms.MidiSongByMeasure):
    """in place operation. TT stands for train/test, because that's where we use it in this project."""
    for t in S.tracks:
        t.transpose_by_octaves_into_range(range_=cs.ACCEPTABLE_NOTE_RANGE_BY_INST_TRAIN_TEST[t.inst],
                                          cleanup_note_duplicates=True)


def _rhythmically_conditioned_str(S: ms.MidiSongByMeasure, track_i: int, measure_i: int,
                                  tail: str,
                                  note_off_treatment: str,
                                  rhythmic_conditioning_type: None or str):
    if rhythmic_conditioning_type == '1d_flattening':
        a_str = instruction_str(0, ENCODING_INSTRUCTION_RHYTHM_PLACEHOLDER).lstrip(';')
        b_str = ''
    elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
        a_str = instruction_str(0, ENCODING_INSTRUCTION_DISTINCT_PITCH_CLASS_MARKER).lstrip(';')
        b_str = instruction_str(0, ENCODING_INSTRUCTION_EXTRA_NOTE_ONSET_MARKER).lstrip(';')
    else:
        raise ValueError(
            f'rhythmic_conditioning_type={rhythmic_conditioning_type} invalid for rhythmically_conditioned_str')

    # note_off_treatment = 'exclude' would probably be easy to add: Just don't add the d: commands, in the same way we
    # don't add them with drum encodings
    if note_off_treatment not in ('duration',):
        raise ValueError(f'Unsupported note_off_treatment: {note_off_treatment}')

    track = S.tracks[track_i]

    MLs = S.get_measure_lengths()
    max_note_length = _get_max_note_length()
    noteidx_tracker = track.get_noteidx_info_dict(measure_lengths=MLs)

    n_notes_by_click = collections.Counter()
    pitch_classes_by_click = collections.defaultdict(set)
    note_lens_by_click = collections.defaultdict(set)
    for n in track.tracks_by_measure[measure_i].note_ons:
        pitch_classes_by_click[n.click].add(n.pitch % 12)
        this_note_length = min(noteidx_tracker[n.noteidx].length, max_note_length)
        note_lens_by_click[n.click].add(this_note_length)
        n_notes_by_click[n.click] += 1
    clicks = list(pitch_classes_by_click.keys())
    clicks.sort()

    instructions = tail.split(';')[1:]
    res = []
    click = 0
    dur = -1
    is_first_instruction = True
    for instruction in instructions:
        i0, i1 = instruction.split(':')

        if i0 == 'd':
            add_chord_encoding_to_res = is_first_instruction
        elif i0 in ('N', 'D'):
            add_chord_encoding_to_res = is_first_instruction
        elif i0 == 'w':
            click += int(i1)
            res.append(instruction)
            add_chord_encoding_to_res = True
        else:
            raise ValueError(f'unknown instruction: {instruction}')

        if add_chord_encoding_to_res:

            # handle the duration encoding for the chord we have just moved to
            new_dur = max(note_lens_by_click[click])
            if new_dur != dur and not track.is_drum:  # drum tracks don't get duration commands
                res.append(f'd:{new_dur}')
            dur = new_dur

            # handle the note encoding for the chord we have just moved to
            if rhythmic_conditioning_type == '1d_flattening':
                res.append(a_str)
            elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
                n_pitch_classes_here = len(pitch_classes_by_click[click])
                n_notes_here = n_notes_by_click[click]
                cur_res_span = [a_str] * n_pitch_classes_here + [b_str] * (n_notes_here - n_pitch_classes_here)
                res.extend(cur_res_span)

        is_first_instruction = False

    if res:
        return ';' + ';'.join(res)
    else:
        return ''


def encode_midisongbymeasure_with_masks(S: "ms.MidiSongByMeasure",
                                        note_off_treatment: str = 'duration',
                                        mask_locations: None or list[tuple[int, int]] = None,
                                        measure_slice: None or tuple[int, int] = None,
                                        include_heads_for_empty_masked_measures: bool = False,
                                        poly_mono_commands: None or dict[tuple[int, int], str] = None,
                                        track_measure_commands: None or dict[tuple[int, int], str] = None,
                                        explicit_rhythmic_conditioning_locations: None or set[tuple[int, int]] = None,
                                        rhythmic_conditioning_type: None or str = None,
                                        return_labels_too: bool = True,
                                        extra_id_st: int = 0,
                                        extra_id_max: int = 255,
                                        velocity_overrides: None or dict[int, int] = None,
                                        commands_at_end: None or dict[int, str] = None,
                                        ) -> tuple[str, str]:
    """
    mask_locations a list of tuples of the form (track_index, measure_index).
    Note that measure indexes are relative to the start of the song, even if measure_slice targets an area of the song
    following the start.

    measure_slice a tuple of the form (start_measure, end_measure).
    Like range() and slicing operations, start_measure is included in the result and end_measure is not.
    Example: To get the string encoding measures 14 and 15, you would set measure_slice = (14, 16).

    include_heads_for_empty_masked_measures should be set to False for fine-tuning, validation, and testing; set it to
    True for inference (e.g., in Reaper). Note that empty unmasked measures NEVER include heads.

    poly_mono_commands a (default)dict with keys of the form (track_index, measure_index) and corresponding values of
    either ';<poly>' or ';<mono>' or ''. ;<poly> and ;<mono> commands will only be inserted for masked
    (track_index, measure_index) pairs. Deprecated as of v2. Use track_measure_commands instead.

    track_measure_commands a (default)dict with keys of the form (track_index, measure_index) and corresponding values
    of strings like '' or ';<instruction_3>;<instruction_4>' etc. Values will only be inserted for masked
    (track_index, measure_index) pairs. Used in the same way as poly_mono_commands and takes precedence over
    poly_mono_commands.

    explicit_rhythmic_conditioning_locations a set with elements of the form (track_index, measure_index). These
    are locations where, if present in mask_locations, we will insert explicit rhythmic conditioning IN PLACE of
    what would normally be encoded. rhythmic_conditioning_type, which can be '1d_flattening' or
    'n_pitch_classes_and_n_notes', tells us how exactly to do so.

    velocity_overrides a dict, *not* a defaultdict, of the form measure_i: avg_vel_of_measure_i.
    This is useful for inference in Reaper when there are no labels and a whole measure is masked.
    Measure indexes not in velocity_overrides use the computed avg velocity from S.

    commands_at_end a (default)dict with keys of the form track_index, and corresponding string values of the form
    '' or ';<instruction_3>' or ';<instruction_3>;<instruction_7>' or similar strings understood by the
    downstream tokenizer. Commands are always added exactly as given for the track_index's supplied,
    regardless of whether anything in those tracks is masked.

    If return_labels is True, the output will be a tuple of the form (input_ids, labels); otherwise, the output
    will be a tuple of the form (input_ids, ''). Use True for fine-tuning, validation, and testing. Use False for
    inference.
    """
    if mask_locations is None:
        mask_locations = []
    mask_locations = copy.copy(mask_locations)  # since we may alter mask_locations below

    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())

    track_measure_commands_temp = collections.defaultdict(str)
    if poly_mono_commands is not None:
        track_measure_commands_temp.update(poly_mono_commands)
    if track_measure_commands is not None:
        track_measure_commands_temp.update(track_measure_commands)
    track_measure_commands = track_measure_commands_temp

    if explicit_rhythmic_conditioning_locations is None:
        explicit_rhythmic_conditioning_locations = set()

    if explicit_rhythmic_conditioning_locations:
        if rhythmic_conditioning_type == '1d_flattening':
            r_cond_str = ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM
        elif rhythmic_conditioning_type == 'n_pitch_classes_and_n_notes':
            r_cond_str = ENCODING_INSTRUCTION_REPLACE_KEEPING_RHYTHM_AND_N_NOTES_AND_N_PITCH_CLASSES
        else:
            raise ValueError(
                f'rhythmic_conditioning_type={rhythmic_conditioning_type} invalid for nonempty explicit rhythmic conditioning locations')

    if velocity_overrides is None:
        velocity_overrides = {}

    commands_at_end_temp = collections.defaultdict(str)
    if commands_at_end is not None:
        commands_at_end_temp.update(commands_at_end)
    commands_at_end = commands_at_end_temp

    measure_lens = S.get_measure_lengths()
    tempos = S.get_tempo_at_start_of_each_measure()

    measure_st_strings = collections.defaultdict(str)

    for measure_i in range(*measure_slice):
        if measure_i in velocity_overrides:
            avg_vel = velocity_overrides[measure_i]
        else:
            avg_vel = _avg_vel_of_measure(S.get_measure(measure_idx=measure_i))
        s = ';M:{}'.format(get_loudness_level(avg_vel))  # M: how loud (scale of 0 to 7)
        s += ';B:{}'.format(get_bpm_level(tempos[measure_i]))  # B: how fast (tempo; scale of 0 to 7)
        s += ';L:{}'.format(measure_lens[measure_i])  # L: how long (number of clicks)
        measure_st_strings[measure_i] = s

    heads, tails = get_string_encoding_matrices(S, note_off_treatment=note_off_treatment, measure_slice=measure_slice)

    # update mask_locations to contain only track-measure tuples that we are REALLY going to mask
    remove_indexes = []
    for i, T in enumerate(mask_locations):
        tail = tails[T]
        if (not tail and not include_heads_for_empty_masked_measures) or not (measure_slice[0] <= T[1] < measure_slice[1]):
            remove_indexes.append(i)
    for i in reversed(remove_indexes):
        mask_locations.pop(i)
    mask_locations_set = set(mask_locations)
    mask_tr_measure_to_index = {T: i for i, T in enumerate(mask_locations)}

    input_ids = ''
    extra_id_int_to_label = {}

    # create the input_ids str, and record info necessary to create the labels str
    for measure_i in range(*measure_slice):
        input_ids += measure_st_strings[measure_i]
        for tr_i in range(len(S.tracks)):
            cur_T = (tr_i, measure_i)
            head = heads[cur_T]
            tail = tails[cur_T]

            # if we are masking this track measure:
            if cur_T in mask_locations_set:
                if tail or include_heads_for_empty_masked_measures:

                    extra_id_int = mask_tr_measure_to_index[cur_T] + extra_id_st

                    extra_id_str = f';<extra_id_{extra_id_int}>'
                    if extra_id_int > extra_id_max:
                        raise ValueError('extra_id_max not large enough or extra_id_st too large')

                    extra_id_int_to_label[extra_id_int] = tail
                    # labels += extra_id_str + tail

                    input_ids += head
                    input_ids += track_measure_commands[cur_T]
                    # insert other start-of-track-measure commands here if applicable

                    # then add mask token
                    input_ids += extra_id_str

                    # then add rhythmic conditioning
                    if cur_T in explicit_rhythmic_conditioning_locations:
                        input_ids += instruction_str(0, r_cond_str)
                        input_ids += _rhythmically_conditioned_str(S=S, track_i=tr_i, measure_i=measure_i, tail=tail,
                                                                   note_off_treatment=note_off_treatment,
                                                                   rhythmic_conditioning_type=rhythmic_conditioning_type)

            else:
                # if this track-measure is not masked, only include it if there is a non-'' tail
                if tail:
                    input_ids += head + tail

    # add commands at end of input_ids str
    end_cmd_s = ''
    for tr_i, tr in enumerate(S.tracks):
        c = commands_at_end[tr_i]
        if c:
            head = heads[(tr_i, measure_slice[0])]
            end_cmd_s += head + c
    if end_cmd_s:
        end_cmd_s = instruction_str(0, ENCODING_INSTRUCTION_INSTRUCTIONS_AT_END_SEP) + end_cmd_s
    input_ids += end_cmd_s

    # create labels str
    labels = ''
    extra_id_ints = sorted(list(extra_id_int_to_label.keys()))
    for extra_id_int in extra_id_ints:
        labels += f';<extra_id_{extra_id_int}>' + extra_id_int_to_label[extra_id_int]

    if return_labels_too:
        return input_ids, labels
    else:
        return input_ids, ''


def encode_midisongbymeasure(S: ms.MidiSongByMeasure, note_off_treatment='duration') -> str:
    """Mainly for testing and pretraining. Template for finetuning."""
    measure_lens = S.get_measure_lengths()
    tempos = S.get_tempo_at_start_of_each_measure()

    measure_st_strings = []
    for measure_i in range(S.get_n_measures()):
        avg_vel = _avg_vel_of_measure(S.get_measure(measure_idx=measure_i))
        s = ';M:{}'.format(get_loudness_level(avg_vel))  # M: how loud (scale of 0 to 7)
        s += ';B:{}'.format(get_bpm_level(tempos[measure_i]))  # B: how fast (tempo; scale of 0 to 7)
        s += ';L:{}'.format(measure_lens[measure_i])  # L: how long (number of clicks)
        measure_st_strings.append(s)

    heads, tails = get_string_encoding_matrices(S, note_off_treatment=note_off_treatment)

    # put it all together
    res = ''
    for measure_i in range(S.get_n_measures()):
        res += measure_st_strings[measure_i]
        for tr_i in range(len(S.tracks)):
            head = heads[(tr_i, measure_i)]
            tail = tails[(tr_i, measure_i)]
            # only include this measure and track in the string if there is a non-'' tail
            if tail:
                res += head + tail
    return res


# test written via test_encode_midisongbymeasure_no_note_offs
def _get_string_encoding_matrices_no_note_offs(
        S: ms.MidiSongByMeasure,
        measure_slice: None or tuple[int, int] = None,
        ) -> tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]:
    """we assume that note on's and note off's are already sorted by click and pitch"""
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;N:Y;w:Z... etc
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice
    for tr_i, tr in enumerate(S.tracks):
        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        s_tail += ';N:{}'.format(n.pitch)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        # note that instrument repetition counters are song-wide, regardless of the measure slice.
        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_duration_commands
def _get_string_encoding_matrices_with_note_duration_commands(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;d:X;N:Y;w:Z... etc (d: means "duration for new notes in this measure")
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice

    MLs = S.get_measure_lengths()
    max_note_length = _get_max_note_length()
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict(measure_lengths=MLs)

        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                cur_length = -1
                # cur_vel = -1
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    # if I wanted to add velocity commands, here's where I'd do it
                    # if vel_bucket(n.vel) != cur_vel:
                    #     s_tail += ';v:{}'.format(vel_bucket(n.vel))
                    #     cur_vel = vel_bucket(n.vel)
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        this_note_length = noteidx_tracker[n.noteidx].length
                        if this_note_length is None:
                            this_note_length = 0
                        length = min(this_note_length, max_note_length)
                        if length != cur_length:
                            s_tail += ';d:{}'.format(length)  # always put "duration" commands right before note on's
                            cur_length = length  # and only add duration commands for non-drum notes
                        s_tail += ';N:{}'.format(n.pitch)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_lengths
def _get_string_encoding_matrices_with_note_lengths(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;N:X:Y;w:Z... etc  # N:X:Y means note pitch X, duration Y.
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice

    MLs = S.get_measure_lengths()
    max_note_length = _get_max_note_length()
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict(measure_lengths=MLs)

        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        this_note_length = noteidx_tracker[n.noteidx].length
                        if this_note_length is None:
                            this_note_length = 0
                        length = min(this_note_length, max_note_length)
                        s_tail += ';N:{}:{}'.format(n.pitch, length)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_offs
# uses ties and explicit note off messages
def _get_string_encoding_matrices_including_note_offs(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    """we assume that note on's and note off's are already sorted by click and pitch"""
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;T:X;N:Y;w:Z;/N:Q... etc
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict()

        # compute where we will have ties for this track
        if not tr.is_drum:
            ties_by_measure = collections.defaultdict(set)
            for idx, info in noteidx_tracker.items():
                if info.measure_note_on < info.measure_note_off:
                    if info.measure_note_off > info.measure_note_on + 1 or info.note_off.click > 0:
                        # then create ties
                        if info.note_off.click == 0:
                            upper = info.measure_note_off
                        else:
                            upper = info.measure_note_off + 1
                        for measure_i in range(info.measure_note_on + 1, upper):
                            ties_by_measure[measure_i].add(info.note_on.pitch)
            # sort ties by pitch
            d = collections.defaultdict(list)
            for measure_i, L in ties_by_measure.items():
                d[measure_i] = sorted(list(L))
            ties_by_measure = d
        else:
            ties_by_measure = collections.defaultdict(list)

        # next, build up our result one measure at a time for this track
        for measure_i, measure_t in enumerate(tr.tracks_by_measure):

            # separate note ons by click and sort by pitch at each click
            note_ons_by_click = collections.defaultdict(list)
            for n in measure_t.note_ons:
                note_ons_by_click[n.click].append(n)

            # separate note offs into "before" and "after" at each click
            note_offs_before_by_click = collections.defaultdict(list)
            note_offs_after_by_click = collections.defaultdict(list)
            if not tr.is_drum:
                for n in measure_t.note_offs:
                    if noteidx_tracker[n.noteidx].measure_note_on == measure_i and noteidx_tracker[n.noteidx].note_on.click == n.click:
                        note_offs_after_by_click[n.click].append(n)
                    else:
                        if n.click != 0:
                            note_offs_before_by_click[n.click].append(n)

            # get our list of clicks for this measure and track
            clicks = set(note_ons_by_click.keys())
            if not tr.is_drum:
                clicks = clicks.union(set(note_offs_before_by_click.keys()))
                clicks = clicks.union(set(note_offs_after_by_click.keys()))
            clicks = sorted(list(clicks))

            # create string for this track and measure
            s_tail = ''
            # handle ties first
            for tie in ties_by_measure[measure_i]:
                s_tail += ';T:{}'.format(tie)
            # build the rest of the string click by click
            prev_click = 0
            for click in clicks:
                if click != prev_click:
                    s_tail += ';w:{}'.format(click - prev_click)
                    prev_click = click
                for n in note_offs_before_by_click[click]:
                    s_tail += ';/N:{}'.format(n.pitch)
                for n in note_ons_by_click[click]:
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        s_tail += ';N:{}'.format(n.pitch)
                for n in note_offs_after_by_click[click]:
                    s_tail += ';/N:{}'.format(n.pitch)

            res_array_tails[(tr_i, measure_i)] = s_tail

            if instrument_repetition_counter[tr.inst]:
                inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
            else:
                inst_rep_str = ''

            s_head = ';I:{}'.format(tr.inst) + inst_rep_str

            res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    # This function is fairly inefficient: Since we need to compute ties, we need to encode the whole song before
    # restricting to the input measure_slice.
    def do_del(heads_or_tails):
        low, high = measure_slice
        to_del = set()
        for k, v in heads_or_tails.items():
            if k[1] < low or k[1] >= high:
                to_del.add(k)
        for k in to_del:
            del heads_or_tails[k]

    if measure_slice is not None:
        do_del(res_array_heads)
        do_del(res_array_tails)

    return res_array_heads, res_array_tails


def _get_max_note_length():
    return 8 * ms.extended_lcm(cs.QUANTIZE)  # 8 QN's max length


def get_string_encoding_matrices(S: "ms.MidiSongByMeasure",
                                 note_off_treatment: str = 'duration',
                                 measure_slice: None or "tuple[int, int]" = None,
                                 ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":

    for tr in S.tracks:
        for tm in tr.tracks_by_measure:
            if tm.notes:
                raise ValueError('cannot encode MidiSongByMeasure to string if any of its tracks T has a ByMeasureTrack'
                                 'in its .tracks_by_measure with .notes. Use only .note_ons and .note_offs instead.')
            for x in tm.note_ons:
                if not hasattr(x, "noteidx") or x.noteidx is None:
                    raise ValueError('cannot encode MidiSongByMeasure to string because some note_on lacks a .noteidx')
            for x in tm.note_offs:
                if not hasattr(x, "noteidx") or x.noteidx is None:
                    raise ValueError('cannot encode MidiSongByMeasure to string because some note_off lacks a .noteidx')

    if note_off_treatment == 'include':
        return _get_string_encoding_matrices_including_note_offs(S, measure_slice=measure_slice)
    elif note_off_treatment == 'exclude':
        return _get_string_encoding_matrices_no_note_offs(S, measure_slice=measure_slice)
    elif note_off_treatment == 'length':
        return _get_string_encoding_matrices_with_note_lengths(S, measure_slice=measure_slice)
    elif note_off_treatment == 'duration':
        return _get_string_encoding_matrices_with_note_duration_commands(S, measure_slice=measure_slice)
    else:
        raise ValueError("note_off_treatment must be one of 'include', 'exclude', 'length', or 'duration'.")
