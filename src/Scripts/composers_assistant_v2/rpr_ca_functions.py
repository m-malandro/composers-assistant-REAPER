import collections
import random
import statistics
import copy

import mytrackviewstuff as mt
import mymidistuff as mm
import myfunctions as mf
import midisong as ms
import nn_str_functions as nns
import constants as cs
import encoding_functions as enc
import tokenizer_functions as tok
import preprocessing_functions as pre
import midi_inst_to_name

DEBUG = False
ALWAYS_TOP_P = True  # True or False: False makes the first attempt for any prompt use greedy decoding
EXCL_TRACK_NAME_SUBSTRINGS = ['ignore', 'IGNORE', 'Ignore']
PERMUTE_MASKS = True

CPQ = ms.extended_lcm(cs.QUANTIZE)


def qns_to_clicks(n_qn):
    return round(CPQ * n_qn)


def transpose_by_octaves_into_range(pitch, inst):
    lo, hi = cs.ACCEPTABLE_NOTE_RANGE_BY_INST_RPR[inst]
    if hi - lo < 11:
        return pitch  # if range is less than an octave, just return the pitch
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def get_inst_from_track_name(track_name: str) -> int:

    # first thing: If the track name starts with a number then a space, take the number as the instrument
    tr_name_split = track_name.split(' ')
    try:
        res = int(tr_name_split[0])
        return res
    except ValueError:
        pass

    # next, the default instrument is 0
    res = 0

    # change the instrument if there are relevant words in track_name
    s = track_name.lower()
    insts = sorted(list(cs.INST_TO_MATCHING_STRINGS_RPR.keys()))
    for inst in insts:
        for s_match in cs.INST_TO_MATCHING_STRINGS_RPR[inst]:
            if s_match in s:
                res = inst
                return res

    return res


def get_velocity_intensity_by_measure(s: str, start_measure: int):
    res = {}
    for i in range(start_measure):
        res[i] = -1
    cur_measure = start_measure
    s = s.split(';')
    for instruction in s:
        if instruction and instruction[0] == 'M':
            res[cur_measure] = int(instruction.split(':')[1])  # instruction is of the form M:intensity
            cur_measure += 1
    return res


def get_avg_vel_in_measure(S, measure_i):
    num = 0
    denom = 0
    for t in S.get_measure(measure_i):
        for n in t.note_ons:
            num += n.vel
            denom += 1
    if denom == 0:
        return None
    else:
        return num / denom


class NNInput(object):
    def __init__(self, nn_input_string, extra_id_to_miditake_and_measure_i, MEs_qn, S, continue_,
                 velocity_intensity_by_measure, start_measure, end_measure, extra_id_to_inst,
                 has_fully_masked_inst, S_vel_track, track_gen_options_by_track_idx,
                 requests_by_track_i_and_measure_i):
        self.nn_input_string = nn_input_string
        self.extra_id_to_miditake_and_measure_i = extra_id_to_miditake_and_measure_i
        self.MEs_qn = MEs_qn
        self.S = S
        self.continue_ = continue_
        self.velocity_intensity_by_measure = velocity_intensity_by_measure
        self.start_measure = start_measure
        self.end_measure = end_measure
        self.extra_id_to_inst = extra_id_to_inst
        self.has_fully_masked_inst = has_fully_masked_inst
        self.S_vel_track = S_vel_track
        self.track_gen_options_by_track_idx = track_gen_options_by_track_idx
        self.requests_by_track_i_and_measure_i = requests_by_track_i_and_measure_i


# helper function
def rpr_track_idx_to_S_track_idx(rpr_track_idx, S):
    for s_idx, t in enumerate(S.tracks):
        if rpr_track_idx in t.extra_info['reaper_track_group']:
            return s_idx


def has_note_in_measure_i(note_list, measure_i):
    for n in note_list:
        if mm.QN_to_measure(n.startQN) == measure_i:
            return True
    return False


def _locate_infiller_global_options_FX_loc() -> int:
    """-1 means not found (or not enabled). Look on monitor FX. Get the first enabled instance."""
    from reaper_python import RPR_GetMasterTrack, RPR_TrackFX_GetEnabled, RPR_TrackFX_GetParam
    tr = RPR_GetMasterTrack(-1)
    n_fx = 100  # search up to 100 FX
    for i in range(0x1000000, 0x1000000 + n_fx):
        if RPR_TrackFX_GetEnabled(tr, i):
            v = RPR_TrackFX_GetParam(tr, i, 0, 0, 0)[0]
            if mf.is_approx(v, 54964317):
                return i
    return -1


def _locate_infiller_track_gen_options_FX_loc(track) -> int:
    """-1 means not found (or not enabled). Get the last enabled instance"""
    res = -1
    from reaper_python import RPR_TrackFX_GetEnabled, RPR_TrackFX_GetParam
    for i in range(mt.get_num_FX_on_track(track)):
    # for i, name in enumerate(mt.get_FX_names_on_track(track)):
        if RPR_TrackFX_GetEnabled(track, i):
            val = RPR_TrackFX_GetParam(track, i, 0, 0, 0)[0]
            if mf.is_approx(val, 349583023) or mf.is_approx(val, 349583024):
                res = i
    return res


class GlobalOptionsObj:
    temperature: float = 1.0

    do_rhythm_conditioning: bool = False
    rhythm_conditioning_type: str or None = None

    do_note_range_conditioning: bool = False
    note_range_conditioning_type: str or None = None

    enc_no_repeat_ngram_size: int = 3
    display_track_to_MIDI_inst: bool = True
    generated_notes_are_selected: bool = True
    display_warnings: bool = True
    variation_alg: int = 0


def get_global_options() -> GlobalOptionsObj:
    from reaper_python import RPR_GetMasterTrack, RPR_TrackFX_GetParam
    res = GlobalOptionsObj()
    fx_loc = _locate_infiller_global_options_FX_loc()
    loc_offset = 1
    if fx_loc != -1:
        t = RPR_GetMasterTrack(-1)

        # temperature
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 0 + loc_offset, 0, 0)
        res.temperature = val

        # rhythm conditioning
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 1 + loc_offset, 0, 0)
        res.do_rhythm_conditioning = val > 0.1
        val = int(val)
        if val == 1:
            res.rhythm_conditioning_type = '1d_flattening'
        elif val == 2:
            res.rhythm_conditioning_type = 'n_pitch_classes_and_n_notes'

        # note range conditioning
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 2 + loc_offset, 0, 0)
        res.do_note_range_conditioning = val > 0.1
        val = int(val)
        if val == 1:
            res.note_range_conditioning_type = 'loose'
        elif val == 2:
            res.note_range_conditioning_type = 'strict'

        # other options
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 3 + loc_offset, 0, 0)
        res.enc_no_repeat_ngram_size = int(val)
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 4 + loc_offset, 0, 0)
        res.variation_alg = int(val)
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 5 + loc_offset, 0, 0)
        res.display_track_to_MIDI_inst = bool(val)
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 6 + loc_offset, 0, 0)
        res.generated_notes_are_selected = bool(val)
        val, _, _, _, _, _ = RPR_TrackFX_GetParam(t, fx_loc, 7 + loc_offset, 0, 0)
        res.display_warnings = bool(val)

    return res


def get_infiller_track_gen_options_by_track_idx() -> "dict[int, dict[str, int]]":
    from reaper_python import RPR_TrackFX_GetParam
    res = {}

    loc_offset = 1

    for i, t in mt.get_tracks_by_idx().items():
        infiller_gen_track_fx_loc = _locate_infiller_track_gen_options_FX_loc(t)
        if infiller_gen_track_fx_loc > -1:
            this_res = {}

            # get rpr_script_min_val. Used to correct -1 vs 0 offset for dropdown menus vs sliders.
            # This was more relevant when there was a slider version of the track-options JSFX, but it is still needed
            # here in the code.
            # min_val will be 0 for the dropdown items
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 13 + loc_offset, 0, 0)
            min_val = Float_retval  # either -1 or 0

            # vert density
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 0 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)  # when min_val is -1, val = Float_retval.
            if val > -1:
                this_res[enc.MEASUREMENT_VERT_NOTE_ONSET_DENSITY] = int(val)

            # vert n pitch classes
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 1 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG] = int(val)

            # horiz density
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 2 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY] = int(val)

            # horiz density diversity
            # Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 3 + loc_offset, 0, 0)
            # val = Float_retval + (-1 * min_val - 1)
            # if val > -1:
            #     this_res[enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE] = int(val)

            # (horiz) rhythmic interest
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 3 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY] = int(val) + 1  # translation from interface to model control

            # step prob
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 4 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.MEASUREMENT_PITCH_STEP_PROB] = int(val)

            # leap prob
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 5 + loc_offset, 0, 0)
            val = Float_retval + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.MEASUREMENT_PITCH_LEAP_PROB] = int(val)

            # lowest note (strict)
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 6 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT] = int(val)

            # highest note (strict)
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 7 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT] = int(val)

            # lowest note (loose)
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 8 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE] = int(val)

            # highest note (loose)
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 9 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res[enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE] = int(val)

            # lowest velocity for new notes
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 10 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res['low_vel'] = int(val)

            # highest velocity for new notes
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 11 + loc_offset, 0, 0)
            val = Float_retval  # + (-1 * min_val - 1)
            if val > -1:
                this_res['high_vel'] = int(val)

            # can be vert copy
            Float_retval, _, _, _, Float_minvalOut, Float_maxvalOut = RPR_TrackFX_GetParam(t, infiller_gen_track_fx_loc, 12 + loc_offset, 0, 0)
            val = Float_retval
            if val == 0:  # the logic on this one is different
                this_res[enc.MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE] = 1

            if this_res:
                res[i] = this_res
    return res


class UserCommandsForNNInputStr:
    commands_at_end: str = ''
    rhythmic_conditioning_locations: set
    track_measure_commands: dict[tuple[int, int], str]
    track_gen_options_by_track_idx: dict[int, dict[str, str]]


def get_user_commands_for_nn_input_str(S, do_note_range_conditioning_by_measure, do_rhythmic_conditioning,
                                       note_range_conditioning_type, requests_by_track_i_and_measure_i,
                                       track_names) -> UserCommandsForNNInputStr:
    # get poly/mono commands
    pmc = collections.defaultdict(str)
    for T in requests_by_track_i_and_measure_i:
        track_i, measure_i = T
        if 'poly' in track_names[track_i].lower():
            S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
            pmc[(S_track_i, measure_i)] = ';<poly>'
        elif 'mono' in track_names[track_i].lower():
            S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
            pmc[(S_track_i, measure_i)] = ';<mono>'
    pmc_by_S_track = collections.defaultdict(list)
    for k, v in pmc.items():
        pmc_by_S_track[k[0]].append(v)

    # now get user-defined track controls for the end of the nn input string...
    requests_by_track_i = set()
    for T in requests_by_track_i_and_measure_i:
        track_i, measure_i = T
        requests_by_track_i.add(track_i)
    track_gen_options_by_track_idx = get_infiller_track_gen_options_by_track_idx()
    if DEBUG:
        print('Track gen options:')
        print(track_gen_options_by_track_idx)
    commands_at_end = collections.defaultdict(str)

    for track_i in requests_by_track_i:
        if track_i in track_gen_options_by_track_idx:
            S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
            # it's probably important to add these commands to the nn input string in the same order we did for training
            for key in [enc.MEASUREMENT_VERT_NOTE_ONSET_DENSITY,
                        enc.MEASUREMENT_VERT_NOTE_ONSET_N_PITCH_CLASSES_AVG,
                        enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY,
                        enc.MEASUREMENT_HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE,
                        enc.MEASUREMENT_HORIZ_NOTE_ONSET_IRREGULARITY,
                        enc.MEASUREMENT_PITCH_STEP_PROB,
                        enc.MEASUREMENT_PITCH_LEAP_PROB,
                        enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT,
                        enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT,
                        enc.ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE,
                        enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE
                        ]:
                try:
                    instruction_str = enc.instruction_str(value=track_gen_options_by_track_idx[track_i][key],
                                                          is_drum=S.tracks[S_track_i].is_drum,
                                                          measurement_str_or_encoding_instruction_str=key)
                except KeyError:
                    instruction_str = ''

                # further "correction" for mono/poly in track names
                if key == enc.MEASUREMENT_VERT_NOTE_ONSET_DENSITY and instruction_str == '':
                    if pmc_by_S_track[S_track_i] and all(x == ';<mono>' for x in pmc_by_S_track[S_track_i]):
                        instruction_str = enc.instruction_str(value=0,
                                                              is_drum=S.tracks[S_track_i].is_drum,
                                                              measurement_str_or_encoding_instruction_str=key)
                    elif pmc_by_S_track[S_track_i] and all(x == ';<poly>' for x in pmc_by_S_track[S_track_i]):
                        instruction_str = enc.instruction_str(value=random.randint(1, 4),
                                                              is_drum=S.tracks[S_track_i].is_drum,
                                                              measurement_str_or_encoding_instruction_str=key)

                # TODO - when doing rhythmic conditioning, only add this instruction str if it makes sense
                commands_at_end[S_track_i] += instruction_str
    # ...and also get "not an octave collapse" commands and note ranges by measure (i.e., track_measure_commands)

    track_measure_commands = collections.defaultdict(str)
    for T in requests_by_track_i_and_measure_i:
        track_i, measure_i = T
        key = enc.MEASUREMENT_THIS_TRACK_MEASURE_IS_NOT_AN_OCTAVE_COLLAPSE_OF_ANY_OTHER_TRACK_IN_THIS_MEASURE
        try:
            instruction_str = enc.instruction_str(track_gen_options_by_track_idx[track_i][key],
                                                  measurement_str_or_encoding_instruction_str=key)
        except KeyError:
            instruction_str = ''
        S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
        track_measure_commands[(S_track_i, measure_i)] += instruction_str

        if do_note_range_conditioning_by_measure:
            pitch_range = S.pitch_range(tr_i=S_track_i, measures=[measure_i])
            if pitch_range is not None:
                lo, hi = pitch_range
                is_drum = S.tracks[S_track_i].is_drum
                if note_range_conditioning_type == 'strict':
                    track_measure_commands[(S_track_i, measure_i)] += enc.instruction_str(lo,
                                                                                          enc.ENCODING_INSTRUCTION_LOWEST_NOTE_STRICT,
                                                                                          is_drum=is_drum)
                    track_measure_commands[(S_track_i, measure_i)] += enc.instruction_str(hi,
                                                                                          enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_STRICT,
                                                                                          is_drum=is_drum)
                elif note_range_conditioning_type == 'loose':
                    track_measure_commands[(S_track_i, measure_i)] += enc.instruction_str(lo,
                                                                                          enc.ENCODING_INSTRUCTION_LOWEST_NOTE_LOOSE,
                                                                                          is_drum=is_drum)
                    track_measure_commands[(S_track_i, measure_i)] += enc.instruction_str(hi,
                                                                                          enc.ENCODING_INSTRUCTION_HIGHEST_NOTE_LOOSE,
                                                                                          is_drum=is_drum)
                else:
                    raise ValueError(f'unrecognized note range conditioning type:{note_range_conditioning_type}')
    rhythmic_conditioning_locations = set()
    if do_rhythmic_conditioning:
        for track_i, measure_i in requests_by_track_i_and_measure_i:
            S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
            if S.tracks[S_track_i].tracks_by_measure[measure_i].note_ons:
                rhythmic_conditioning_locations.add((S_track_i, measure_i))
    if DEBUG:
        print('rhythmic conditioning locations (S track indexes):')
        print(rhythmic_conditioning_locations)
    res = UserCommandsForNNInputStr()
    res.commands_at_end = commands_at_end
    res.rhythmic_conditioning_locations = rhythmic_conditioning_locations
    res.track_measure_commands = track_measure_commands
    res.track_gen_options_by_track_idx = track_gen_options_by_track_idx

    return res


def get_nn_input_from_project(warn_if_no_masks=True, mask_empty_midi_items=True, mask_selected_midi_items=False,
                              do_rhythmic_conditioning=False,
                              rhythmic_conditioning_type='1d_flattening',
                              do_note_range_conditioning_by_measure=False,
                              note_range_conditioning_type='loose',
                              display_track_to_MIDI_inst=True,
                              display_warnings=True):

    if rhythmic_conditioning_type is not None:
        mask_selected_midi_items = True

    WARNINGS_PRINTED = set()
    notes_by_trackidx = mt.get_all_visible_midi_notes_by_trackidx(attach_parent_take_to_each_note=False,
                                                                  include_muted_items=False,
                                                                  include_muted_tracks=False,
                                                                  restrict_to_time_selection=False,
                                                                  excl_track_name_substrings=EXCL_TRACK_NAME_SUBSTRINGS,
                                                                  display_warnings=display_warnings)
    track_names = mm.get_track_names()
    AT_by_track_idx = mt.get_active_takes_with_info_by_trackidx(include_muted_tracks=False, midi_items_only=True,
                                                                include_muted_items=False,
                                                                excl_track_name_substrings=EXCL_TRACK_NAME_SUBSTRINGS)

    infiller_vel_track_idxs = [i for i, name in enumerate(track_names) if 'vel_track' in name.lower()]

    # define start_measure and end_measure based on time selection and notes in project.
    start_measure = 0
    time_selection = mt.get_time_selection()
    if time_selection is not None:
        start_measure = mt.qn_to_measure(mt.sec_to_QN(time_selection[0])+.0001)

    if time_selection is not None:
        end_sel = mt.qn_to_measure(mt.sec_to_QN(time_selection[1])-.0001)
        end_measure = end_sel
    else:
        end_measure = None

    if DEBUG:
        print('start_measure: {}'.format(start_measure))
        print('end_measure: {}'.format(end_measure))

    # consolidate tracks by looking for "Child" in the track names
    track_groupings = []
    for i, t in enumerate(track_names):
        if i:
            if 'Child' in t:
                track_groupings[-1].append(i)
            else:
                track_groupings.append([i])
        else:
            track_groupings.append([i])

    # make note of all requests: The convention is that empty items create requests for all of their measures, while
    # When doing rhythmic conditioning, items with notes create requests only for the measures in which they hold notes.
    # We are not imposing any ORDER on the requests yet.
    # requests_by_track_i_and_measure_i tells us both what our masks are and which take corresponds to each mask;
    # however, these are reaper track indexes, NOT MidiSongByMeasure track indexes.
    requests_by_track_i_and_measure_i = {}
    for track_i, L in AT_by_track_idx.items():
        if track_i not in infiller_vel_track_idxs:  # vel guide tracks don't make requests.
            # first, get the track group of this track
            this_track_group = [track_i]
            for track_group in track_groupings:
                if track_i in track_group:
                    this_track_group = track_group
                    break

            # then add requests for this track (group)
            for take_w_info in L:
                notes = mm.get_notes(take=take_w_info.take)
                have_at_least_one_request = False
                if mask_empty_midi_items and not notes and not mm.get_CCs(take=take_w_info.take):  # then we have at least one request here
                    have_at_least_one_request = True
                if mask_selected_midi_items and take_w_info.is_selected:
                    have_at_least_one_request = True
                if have_at_least_one_request:
                    st_measure_req = mt.qn_to_measure(take_w_info.start_QN + .0001)
                    st_measure_req = max(st_measure_req, start_measure)  # bump up the first measure of the request based on time sel
                    end_measure_req = mt.qn_to_measure(take_w_info.end_QN_in_track - .0001)
                    if end_measure is not None:
                        end_measure_req = min(end_measure_req, end_measure)  # lower the last measure of the request based on time sel
                    for measure_i in range(st_measure_req, end_measure_req + 1):
                        have_request_here = False
                        # is_empty_in_this_measure says whether the entire track group is empty in this measure.
                        is_empty_in_this_measure = all(not has_note_in_measure_i(notes_by_trackidx.get(t, []), measure_i) for t in this_track_group)
                        # we have a request here if there are no notes in this measure in the whole track group...
                        if mask_empty_midi_items and is_empty_in_this_measure:
                            have_request_here = True
                        # ...or if this item is selected
                        if mask_selected_midi_items and take_w_info.is_selected:
                            have_request_here = True
                            # ...but when doing rhythmic conditioning, only if this track-measure has note-ons
                            if rhythmic_conditioning_type is not None:
                                if not has_note_in_measure_i(notes_by_trackidx.get(track_i, []), measure_i):
                                    have_request_here = False
                            # or if the item is completely empty of notes
                                if not mm.get_notes(take_w_info.take, unmuted_only=True) and mask_empty_midi_items and is_empty_in_this_measure:
                                    have_request_here = True
                        if have_request_here:
                            requests_by_track_i_and_measure_i[(track_i, measure_i)] = take_w_info


    if end_measure is None:
        # then end_measure will be the last measure that has a note or a request in it
        maxqn = -1
        for i, note_L in notes_by_trackidx.items():
            for note in note_L:
                maxqn = max(maxqn, note.endQN)
        max_measure = mt.qn_to_measure(maxqn - .0001)
        for (track_i, measure_i) in requests_by_track_i_and_measure_i:
            max_measure = max(max_measure, measure_i)

        end_measure = max_measure

    # Get all measure endpoints from the 0th measure to the last measure
    MEs_qn = mt.get_measure_endpoints_in_QN(first_measure_index=0,
                                            last_measure_index=end_measure + 4)  # add a few measures for note offs at end and such

    MEs_click_960 = [round(x * 960) for x in MEs_qn]
    if DEBUG:
        print('MEs qn', MEs_qn)

    # Add all notes in selection to a ms.MidiSong with cpq. Include only tracks that have notes and/or requests.
    tracks = []
    vel_tracker_tracks = []
    for track_group in track_groupings:
        primary_track_name = track_names[track_group[0]]
        T = ms.Track(inst=get_inst_from_track_name(primary_track_name), name=primary_track_name)
        T.extra_info['reaper_track_group'] = track_group
        for track_idx in track_group:
            if track_idx in notes_by_trackidx:
                note_L = notes_by_trackidx[track_idx]
            else:
                note_L = []
            for note in note_L:
                # only include notes with onsets in the time selection; no need to include everything
                note_st_click_960 = round(note.startQN * 960)
                note_end_click_960 = round(note.endQN * 960)
                if MEs_click_960[0] <= note_st_click_960+.0001 and note_end_click_960+.0001 < MEs_click_960[-1]:
                    T.notes.append(ms.Note(pitch=note.pitch,
                                           vel=note.vel,
                                           click=note_st_click_960,
                                           end=note_end_click_960))
        # if this track group has an unmuted midi item on it anywhere in the project, add the corresponding track to S.
        # Note that AT_by_track_idx only has tracks that are themselves unmuted.
        if any(any(not k.is_muted for k in AT_by_track_idx[t]) for t in track_group):
            if 'vel_track' in primary_track_name.lower():
                vel_tracker_tracks.append(T)
            else:
                tracks.append(T)

    tempo_changes = []
    for measure_idx, me_qn in enumerate(MEs_qn):
        tempo = mt.get_tempo_at_st_of_measure(measure_idx)
        tempo_changes.append(ms.TempoChange(val=tempo, click=round(960 * me_qn)))

    S = ms.MidiSong(cpq=960, tempo_changes=tempo_changes)
    for t in tracks:
        S.tracks.append(t)

    S.sort_tracks_by_inst()  # not by avg pitch; that's assumed by how the user has set up their project
    pre.apply_simplified_drum_map(S)

    S = ms.MidiSongByMeasure.from_MidiSong(S, measure_endpoints=MEs_click_960,
                                           consume_calling_song=True)

    S.quantize_notes_by_measure(q=cs.QUANTIZE)
    S.change_cpq(CPQ)
    enc.transpose_into_acceptable_ranges_TT(S)
    for t in S.tracks:
        t.sort()

    if display_track_to_MIDI_inst:
        display_track_to_inst_dict = {}
        for t in S.tracks:
            if len(t.extra_info["reaper_track_group"]) > 1:
                str_st = f'Tracks {[x + 1 for x in t.extra_info["reaper_track_group"]]}'
                i = min([x + 1 for x in t.extra_info["reaper_track_group"]])
            elif len(t.extra_info["reaper_track_group"]) == 1:
                str_st = f'Track {t.extra_info["reaper_track_group"][0] + 1}'
                i = t.extra_info["reaper_track_group"][0] + 1
            else:
                str_st = f'Track []'
                i = -1
            this_str = f'{str_st} name = {t.name}, midi_inst = {t.inst} ({midi_inst_to_name.midi_inst_to_name[t.inst]})'
            display_track_to_inst_dict[i] = this_str
        for i in sorted(list(display_track_to_inst_dict.keys())):
            print(display_track_to_inst_dict[i])

    # Also set up a MidiSongByMeasure for user velocity imputation
    S_vel_track = ms.MidiSong(cpq=960, tempo_changes=[copy.copy(x) for x in tempo_changes])
    for t in vel_tracker_tracks:
        S_vel_track.tracks.append(t)
    S_vel_track.tracks = [ms.combine_tracks(S_vel_track.tracks, inst_for_result=None)]  # combine to exactly one track
    S_vel_track = ms.MidiSongByMeasure.from_MidiSong(S_vel_track, measure_endpoints=MEs_click_960,
                                                     consume_calling_song=True)
    S_vel_track.quantize_notes_by_measure(q=cs.QUANTIZE)
    S_vel_track.change_cpq(CPQ)
    for t in S_vel_track.tracks:
        t.sort()

    if DEBUG:
        print('extra infos:')
        for t in S.tracks:
            print(t.extra_info)

    # change our requests into masks that make sense to S
    masks = []
    S_mask_to_rpr_mask = {}
    for T in requests_by_track_i_and_measure_i:
        track_i, measure_i = T
        S_track_i = rpr_track_idx_to_S_track_idx(track_i, S)
        if start_measure <= measure_i <= end_measure:
            if len(masks) < 256:
                masks.append((S_track_i, measure_i))
                S_mask_to_rpr_mask[(S_track_i, measure_i)] = (track_i, measure_i)
            else:
                warning = 'NEURAL NET WARNING: You have requested that more than 256 measures be filled. '
                warning += 'The neural net can only fill 256 measures at a time. '
                warning += 'Please run this script again to fill more measures.'
                if warning not in WARNINGS_PRINTED and display_warnings:
                    print(warning)
                    WARNINGS_PRINTED.add(warning)

    if not masks and warn_if_no_masks:
        warning = 'ERROR: No requests made. '
        if mask_empty_midi_items and not mask_selected_midi_items:
            warning += 'Please include at least one empty midi item in your time selection and try again. '
            warning += 'Your empty midi item must cover at least one measure. '
            warning += 'This script will only write notes to measures in empty midi items.'
        elif mask_empty_midi_items and mask_selected_midi_items:
            warning += 'Please include at least one empty or selected midi item in your time selection and try again. '
            warning += 'Such a midi item must cover at least one measure. '
            warning += 'This script will only write notes to measures in selected or empty midi items.'
        elif not mask_empty_midi_items and mask_selected_midi_items:
            warning += 'Please include at least one selected midi item in your time selection and try again. '
            warning += 'Such a midi item must cover at least one measure. '
            warning += 'This script will only write notes to measures in selected midi items.'
        mt.messagebox(msg=warning, title='No requests in selection', int_type=0)

    # encode S to nn string
    if masks:
        extra_id_st = random.randint(0, 255 - (len(masks) - 1))
        continue_ = True
    else:
        extra_id_st = 0
        continue_ = False

    # record extra_id to miditake and measure_i
    masks_for_S_in_order = sorted(list(masks), key=lambda tup: (tup[1], tup[0]))
    if PERMUTE_MASKS:
        masks_for_S_in_order = random.sample(masks_for_S_in_order, len(masks_for_S_in_order))  # random permutation
    if DEBUG:
        print('masks:')
        print(masks_for_S_in_order)

    extra_id_to_miditake_and_measure_i = {}
    extra_id_to_inst = {}
    for i, T in enumerate(masks_for_S_in_order):
        rpr_track_i, measure_i = S_mask_to_rpr_mask[T]
        extra_id_to_miditake_and_measure_i[extra_id_st + i] = (requests_by_track_i_and_measure_i[(rpr_track_i, measure_i)], measure_i)
        # also record the instrument that goes with each extra_id
        extra_id_to_inst[extra_id_st + i] = get_inst_from_track_name(S.tracks[T[0]].name)
        if DEBUG:
            print('eid to inst:', extra_id_to_inst)

    # next, use vel_track track to impute velocity requests
    velocity_overrides = {}
    empty_measures_in_S_vel_track = S_vel_track.get_measure_indexes_containing_no_note_ons()
    vel_track_velocity_overrides = {}
    for measure_i in empty_measures_in_S_vel_track:
        # if there is no vel info given, default to dynamics level 5
        vel_track_velocity_overrides[measure_i] = enc.DYNAMICS_DEFAULTS[5]
    s_vel_track, _ = enc.encode_midisongbymeasure_with_masks(S_vel_track,
                                                             note_off_treatment=tok.spm_type_to_note_off_treatment(cs.SPM_TYPE),
                                                             mask_locations=None,
                                                             measure_slice=(start_measure, end_measure + 1),
                                                             include_heads_for_empty_masked_measures=False,
                                                             poly_mono_commands=None,
                                                             return_labels_too=False,
                                                             extra_id_st=0,
                                                             extra_id_max=255,
                                                             velocity_overrides=vel_track_velocity_overrides,
                                                             )
    vel_intensity_by_measure_for_vel_track = get_velocity_intensity_by_measure(s_vel_track, start_measure=start_measure)

    empty_measures_in_S_vel_track_set = set(empty_measures_in_S_vel_track)
    for measure_i in range(S.get_n_measures()):
        encoded_vel = vel_intensity_by_measure_for_vel_track.get(measure_i, 5)
        if encoded_vel >= 0 and measure_i not in empty_measures_in_S_vel_track_set:
            velocity_overrides[measure_i] = enc.DYNAMICS_DEFAULTS[encoded_vel]

    user_commands = get_user_commands_for_nn_input_str(S=S,
                                                       do_note_range_conditioning_by_measure=do_note_range_conditioning_by_measure,
                                                       do_rhythmic_conditioning=do_rhythmic_conditioning,
                                                       note_range_conditioning_type=note_range_conditioning_type,
                                                       requests_by_track_i_and_measure_i=requests_by_track_i_and_measure_i,
                                                       track_names=track_names)

    # now encode S to string
    s, _ = enc.encode_midisongbymeasure_with_masks(S,
                                                   note_off_treatment=tok.spm_type_to_note_off_treatment(cs.SPM_TYPE),
                                                   mask_locations=masks_for_S_in_order,
                                                   measure_slice=(start_measure, end_measure + 1),
                                                   include_heads_for_empty_masked_measures=True,
                                                   track_measure_commands=user_commands.track_measure_commands,
                                                   explicit_rhythmic_conditioning_locations=user_commands.rhythmic_conditioning_locations,
                                                   rhythmic_conditioning_type=rhythmic_conditioning_type,
                                                   return_labels_too=False,
                                                   extra_id_st=extra_id_st,
                                                   extra_id_max=255,
                                                   velocity_overrides=velocity_overrides,
                                                   commands_at_end=user_commands.commands_at_end
                                                   )

    velocity_intensity_by_measure = get_velocity_intensity_by_measure(s, start_measure)
    if DEBUG:
        print('vel overrides:', velocity_overrides)
        print('vel intensity by measure:', velocity_intensity_by_measure)

    # TODO: This logic is not correct anymore if mask_selected_midi_items = True. However, this value is not currently used anywhere else.
    has_fully_masked_inst = any(not t.has_notes() for t in S.tracks)

    return NNInput(nn_input_string=s,
                   extra_id_to_miditake_and_measure_i=extra_id_to_miditake_and_measure_i,
                   MEs_qn=MEs_qn,
                   S=S,
                   continue_=continue_,
                   velocity_intensity_by_measure=velocity_intensity_by_measure,
                   start_measure=start_measure,
                   end_measure=end_measure,
                   extra_id_to_inst=extra_id_to_inst,
                   has_fully_masked_inst=has_fully_masked_inst,
                   S_vel_track=S_vel_track,
                   track_gen_options_by_track_idx=user_commands.track_gen_options_by_track_idx,
                   requests_by_track_i_and_measure_i=requests_by_track_i_and_measure_i
                   )


def instruction_list_to_MyNote_dict(instruction_list, take_with_info, measure_start_qn, measure_end_qn, measure_i, inst, dict_to_update=None,
                                    notes_are_selected=True, track_gen_options_by_track_idx=None):
    if tok.spm_type_to_note_off_treatment(cs.SPM_TYPE) not in ('duration', 'exclude'):
        raise NotImplemented('Writing instruction strings of type "duration" and "exclude" are'
                             ' the only types currently implemented')

    if dict_to_update is None:
        dict_to_update = {}

    if track_gen_options_by_track_idx is None:
        track_gen_options_by_track_idx = {}

    # default velocities for new notes
    def get_vel(qn_in_measure, timesig_numerator=4, timesig_denominator=4, low_vel=105, high_vel=115):
        # return 110  # super override

        vels = {0: low_vel,
                1: round(2*low_vel/3 + high_vel/3),
                2: round(low_vel/3 + 2*high_vel/3),
                3: high_vel}

        if timesig_numerator % 3 == 0 and timesig_denominator % 8 == 0:  # eg, 3/8, 3/16, 6/8, etc
            if mf.is_approx(qn_in_measure, 0):
                return vels[3]
            elif mf.is_approx((qn_in_measure + .00001) % 1.5, 0, .0001):
                return vels[2]
            elif mf.is_approx((qn_in_measure + .00001) % 0.75, 0, .0001):
                return vels[1]
            else:
                return vels[0]

        else:  # pattern for all other time signatures
            if mf.is_approx(qn_in_measure, 0):
                return vels[3]
            elif mf.is_approx(qn_in_measure, 1):
                return vels[1]
            elif mf.is_approx(qn_in_measure, 2):
                return vels[2]
            elif mf.is_approx(qn_in_measure, 3):
                return vels[1]
            elif mf.is_approx(qn_in_measure, 4):
                return vels[3]
            elif mf.is_approx(qn_in_measure, 5):
                return vels[1]
            elif mf.is_approx(qn_in_measure, 6):
                return vels[2]
            elif mf.is_approx(qn_in_measure, 7):
                return vels[1]
            else:
                return vels[0]

    this_track_gen_options = track_gen_options_by_track_idx.get(take_with_info.track_idx, {})
    low_vel = this_track_gen_options.get('low_vel', 105)
    high_vel = this_track_gen_options.get('high_vel', 115)
    measure_end_ppq = mm.QN_to_ppq(x_in_QN=measure_end_qn, take=take_with_info.take)
    ts_numerator, ts_denominator = mt.get_time_sig_at_st_of_measure(measure_idx=measure_i)
    cur_qn_duration = 0.5  # default
    cur_qn_in_measure = 0.0
    cur_noteidx = max(dict_to_update.keys()) + 1 if dict_to_update else 0
    # get notes to write from instruction_list
    pitches_starting_at_pos = collections.defaultdict(set)

    is_selected = 1 if notes_are_selected else 0
    for instruction in instruction_list:
        if instruction[0] in ('w', 'd', 'N', 'D'):  # only write w, d, N, D instructions to Reaper project
            s = instruction.split(':')
            if s[0] == 'd':
                cur_qn_duration = int(s[1]) / CPQ
            elif s[0] == 'w':
                cur_qn_in_measure += int(s[1]) / CPQ
            else:
                pitch = int(s[1])
                note_start_qn = measure_start_qn + cur_qn_in_measure
                # only write a note if it starts in its measure
                if note_start_qn < measure_end_qn:
                    startppq = mm.QN_to_ppq(x_in_QN=note_start_qn, take=take_with_info.take)
                    endppq = mm.QN_to_ppq(x_in_QN=note_start_qn + cur_qn_duration, take=take_with_info.take)
                    # don't let drum hits hang over into the next measure
                    if s[0] == 'D':
                        endppq = min(endppq, measure_end_ppq)
                    base_vel = get_vel(qn_in_measure=cur_qn_in_measure,
                                       timesig_numerator=ts_numerator,
                                       timesig_denominator=ts_denominator,
                                       low_vel=low_vel,
                                       high_vel=high_vel)
                    note = mm.MyNote(take=take_with_info.take,
                                     noteidx=cur_noteidx,
                                     selected=is_selected,
                                     muted=0,
                                     startppqpos=startppq,
                                     endppqpos=endppq,
                                     chan=0,
                                     pitch=transpose_by_octaves_into_range(pitch, inst),
                                     vel=base_vel)
                    if note.pitch not in pitches_starting_at_pos[note.startppqpos]:
                        dict_to_update[cur_noteidx] = note
                        cur_noteidx += 1
                    pitches_starting_at_pos[note.startppqpos].add(note.pitch)
        else:
            if DEBUG:
                print('NOTICE: unhandled instruction in nn output string: {}'.format(instruction))

    return dict_to_update


def write_dict_to_project(take, to_write):
    # fix note overlaps (if any) in to_write
    mm.correct_note_overlaps(d=to_write)
    # write notes to project
    to_write = list(to_write.values())
    mm.insert_MyNotes(take=take, notes=to_write, noSortInOptional=True)
    mm.MIDI_sort(take=take)


def write_nn_output_to_project(nn_output: str, nn_input_obj, notes_are_selected=True, use_vels_from_tr_measures=False):
    extra_id_to_miditake_and_measure_i = nn_input_obj.extra_id_to_miditake_and_measure_i
    MEs_qn = nn_input_obj.MEs_qn

    def _adjust_vels(take_w_info, measure_i, measure_start_qn):
        notes_by_idx = mm.get_notes(take_w_info.take)
        action_idxs = set(note.noteidx for note in notes_by_idx.values() if mt.qn_to_measure(note.startQN+.0001) == measure_i)

        # Use the velocity track of the nn_input_obj...
        S_vel_for_this_measure = nn_input_obj.S_vel_track.tracks[0].tracks_by_measure[measure_i]

        # ...unless S itself has note on's in this track measure and we're doing rhythmic conditioning -
        # in that case, use those velocities
        if use_vels_from_tr_measures:
            S_track_idx = rpr_track_idx_to_S_track_idx(take_w_info.track_idx, nn_input_obj.S)
            S_vel_2 = nn_input_obj.S.tracks[S_track_idx].tracks_by_measure[measure_i]
            S_vel_for_this_measure = S_vel_2 if S_vel_2.note_ons else S_vel_for_this_measure

        if S_vel_for_this_measure.note_ons:
            vels_by_position = collections.defaultdict(list)
            for n in S_vel_for_this_measure.note_ons:
                vels_by_position[n.click].append(n.vel)
            for click, L in vels_by_position.items():
                vels_by_position[click] = statistics.mean(L)
            vels_by_position = dict(vels_by_position)
            vel_clicks = sorted(list(vels_by_position.keys()))
            click_to_project_QN = {click: measure_start_qn + click / CPQ for click in vels_by_position}
            vel_QNs = [click_to_project_QN[x] for x in vel_clicks]

            new_vels_by_note_idx = {}
            for idx in action_idxs:
                vel_idx = mf.index_of_closest_element_in_sorted_numeric_list(vel_QNs, notes_by_idx[idx].startQN)
                new_vel = vels_by_position[vel_clicks[vel_idx]]
                new_vels_by_note_idx[idx] = new_vel

            for note in notes_by_idx.values():
                if note.noteidx in new_vels_by_note_idx:
                    mm.set_note_properties(take=take_w_info.take, MyNote_instance=note,
                                           set_vel=new_vels_by_note_idx[note.noteidx],
                                           noSortInOptional=True)

    instructions_by_extra_id = nns.instructions_by_extra_id(nn_output)
    # if DEBUG:
    #     print('instructions to write:')
    #     for k, v in instructions_by_extra_id.items():
    #         print(k, ':', v)

    dicts_to_write = collections.defaultdict(dict)
    for extra_id, (miditake_w_info, measure_i) in extra_id_to_miditake_and_measure_i.items():
        dict_to_update = dicts_to_write[miditake_w_info.take]
        instruction_list_to_MyNote_dict(instruction_list=instructions_by_extra_id['<extra_id_{}>'.format(extra_id)],
                                        take_with_info=miditake_w_info,
                                        measure_start_qn=MEs_qn[measure_i],
                                        measure_end_qn=MEs_qn[measure_i + 1],
                                        measure_i=measure_i,
                                        inst=nn_input_obj.extra_id_to_inst[extra_id],
                                        dict_to_update=dict_to_update,
                                        notes_are_selected=notes_are_selected,
                                        track_gen_options_by_track_idx=nn_input_obj.track_gen_options_by_track_idx)
        # also delete any notes in this track-measure if necessary
        eps = 0.0001
        notes_to_del = mm.get_notes(take=miditake_w_info.take, QN_window=[MEs_qn[measure_i]-eps, MEs_qn[measure_i + 1]-eps], unmuted_only=True)
        mm.delete_notes_by_idx(take=miditake_w_info.take, L=notes_to_del)

    for take, write_dict in dicts_to_write.items():
        if DEBUG:
            print('writing to take', take)
        write_dict_to_project(take, write_dict)

    for extra_id, (miditake_w_info, measure_i) in extra_id_to_miditake_and_measure_i.items():
        _adjust_vels(take_w_info=miditake_w_info, measure_i=measure_i, measure_start_qn=MEs_qn[measure_i])


# def get_poly_mono_commands(S: "ms.MidiSongByMeasure", mask: "set[tuple[int, int]]"):
#     poly_mono_commands = collections.defaultdict(str)
#     for mask_tuple in mask:
#         tr_i, m_i = mask_tuple
#         if not S.tracks[tr_i].is_drum:  # drum tracks do not get poly/mono commands
#             if S.is_poly(track_idx=tr_i, measure_idx=m_i):
#                 poly_mono_commands[(tr_i, m_i)] = ';<poly>'
#             else:
#                 poly_mono_commands[(tr_i, m_i)] = ';<mono>'
#     return poly_mono_commands

def find_take_w_info(measure_i, track_group, AT_by_track_idx):
    for t_i in track_group:
        for AT in AT_by_track_idx[t_i]:
            if mt.qn_to_measure(AT.start_QN + .0001) <= measure_i <= mt.qn_to_measure(AT.end_QN_in_track - .0001):
                return AT
    return None


# TODO - add auto-computed user controls...maybe
def create_and_write_variation_for_time_selection(n_cycles=2, mask_pattern_type=0,
                                                  create_variation_only_for_selected_midi_items=False,
                                                  new_notes_are_selected=True,
                                                  temperature=1.0,
                                                  do_rhythmic_conditioning=False,
                                                  rhythmic_conditioning_type='1d_flattening',
                                                  do_note_range_conditioning_by_measure=False,
                                                  note_range_conditioning_type='loose',
                                                  display_track_to_MIDI_inst=True,
                                                  display_warnings=True):
    """return True iff it did anything"""

    # time_selection = mt.get_time_selection()
    # if time_selection is None:
    #     mt.messagebox(msg="Please make a time selection, then run this script again.",
    #                   title="Variation writer error: no time selection",
    #                   int_type=0)
    #     return False

    AT_by_track_idx = mt.get_active_takes_with_info_by_trackidx(include_muted_tracks=False, midi_items_only=True,
                                                                include_muted_items=False,
                                                                excl_track_name_substrings=EXCL_TRACK_NAME_SUBSTRINGS)

    # setup
    nn_input = get_nn_input_from_project(warn_if_no_masks=False, display_track_to_MIDI_inst=display_track_to_MIDI_inst,
                                         display_warnings=display_warnings)
    S = nn_input.S

    act_L = []
    track_measure_to_take_w_info = {}
    for track_i, track in enumerate(S.tracks):
        for measure_i, m_track in enumerate(track.tracks_by_measure):
            if nn_input.start_measure <= measure_i <= nn_input.end_measure:
                if m_track.note_ons and (create_variation_only_for_selected_midi_items or 'fix' not in track.name.lower()):
                    take_w_info = find_take_w_info(measure_i, track.extra_info['reaper_track_group'], AT_by_track_idx)
                    is_selected = take_w_info.is_selected if take_w_info else False
                    is_selected_for_rewrite = True if not create_variation_only_for_selected_midi_items else is_selected
                    if is_selected_for_rewrite:
                        act_L.append((track_i, measure_i))
                        track_measure_to_take_w_info[(track_i, measure_i)] = take_w_info
    if DEBUG:
        print('act_L: {}'.format(act_L))

    measures_act = set(T[1] for T in act_L)
    tracks_act = set(T[0] for T in act_L)

    # create masks
    if mask_pattern_type == 0:  # random (track, measure) pairs
        n_replacements_per_cycle = len(act_L) // n_cycles
        n_replacements = [n_replacements_per_cycle for _ in range(n_cycles)]
        for i in range(len(act_L) % n_cycles):
            n_replacements[i] += 1
        # example: If n_cycles = 3 and len(act_L) = 11, then n_replacements = [4, 4, 3]
        if not n_replacements:
            return False

        if n_replacements[0] > 256:
            # then we will have to do more than n_cycles worth of replacements
            q, r = divmod(len(act_L), 256)
            n_replacements = [256 for _ in range(q)] + [r]

        masks_list = []
        for i, n_r in enumerate(n_replacements):
            masks = random.sample(act_L, n_r)
            masks.sort(key=lambda T: (T[1], T[0]))
            if PERMUTE_MASKS:
                masks = random.sample(masks, len(masks))  # random permutation
            masks_list.append(masks)
            for x in masks:
                i = act_L.index(x)
                act_L.pop(i)

    elif mask_pattern_type == 1:  # random measures
        measures = sorted(list(set(T[1] for T in act_L)))
        if DEBUG:
            print('measures to mask across all cycles: {}'.format(measures))
        n_measures_to_mask_per_cycle = len(measures) // n_cycles
        n_measures_to_mask = [n_measures_to_mask_per_cycle for _ in range(n_cycles)]
        for i in range(len(measures) % n_cycles):
            n_measures_to_mask[i] += 1
        if not n_measures_to_mask:
            return False

        masks_list = []
        for i, n_r in enumerate(n_measures_to_mask):
            masked_measures = set(random.sample(measures, n_r))
            masks = [T for T in act_L if T[1] in masked_measures]
            masks.sort(key=lambda T: (T[1], T[0]))
            if PERMUTE_MASKS:
                masks = random.sample(masks, len(masks))  # random permutation
            # failsafe:
            if len(masks) > 256:
                if len(tracks_act) > 256:
                    return create_and_write_variation_for_time_selection(n_cycles=n_cycles, mask_pattern_type=0)
                else:
                    return create_and_write_variation_for_time_selection(n_cycles=n_cycles + 1, mask_pattern_type=1)
            else:
                masks_list.append(masks)

            for x in masks:
                i = act_L.index(x)
                act_L.pop(i)
            for x in masked_measures:
                i = measures.index(x)
                measures.pop(i)

    elif mask_pattern_type == 2:  # random instruments
        tracks = sorted(list(set(T[0] for T in act_L)))
        n_tracks_to_mask_per_cycle = len(tracks) // n_cycles
        n_tracks_to_mask = [n_tracks_to_mask_per_cycle for _ in range(n_cycles)]
        for i in range(len(tracks) % n_cycles):
            n_tracks_to_mask[i] += 1
        if not n_tracks_to_mask:
            return False

        masks_list = []
        for i, n_r in enumerate(n_tracks_to_mask):
            masked_tracks = set(random.sample(tracks, n_r))
            masks = [T for T in act_L if T[0] in masked_tracks]
            masks.sort(key=lambda T: (T[1], T[0]))
            if PERMUTE_MASKS:
                masks = random.sample(masks, len(masks))  # random permutation
            # failsafe:
            if len(masks) > 256:
                if len(measures_act) > 256:
                    return create_and_write_variation_for_time_selection(n_cycles=n_cycles, mask_pattern_type=0)
                else:
                    return create_and_write_variation_for_time_selection(n_cycles=n_cycles + 1, mask_pattern_type=2)
            else:
                masks_list.append(masks)

            for x in masks:
                i = act_L.index(x)
                act_L.pop(i)
            for x in masked_tracks:
                i = tracks.index(x)
                tracks.pop(i)

    else:
        raise ValueError('mask_pattern_type {} not recognized.'.format(mask_pattern_type))

    track_names = mm.get_track_names()

    for masks in masks_list:
        if DEBUG:
            print('to_mask', masks)

        extra_id_st = random.randint(0, 255 - (len(masks) - 1))
        extra_id_to_mask = {}
        for i, mask in enumerate(masks):
            extra_id_to_mask['<extra_id_{}>'.format(extra_id_st + i)] = mask

        user_commands = get_user_commands_for_nn_input_str(S=S,
                                                           do_note_range_conditioning_by_measure=do_note_range_conditioning_by_measure,
                                                           do_rhythmic_conditioning=do_rhythmic_conditioning,
                                                           note_range_conditioning_type=note_range_conditioning_type,
                                                           requests_by_track_i_and_measure_i=nn_input.requests_by_track_i_and_measure_i,
                                                           track_names=track_names)

        s, _ = enc.encode_midisongbymeasure_with_masks(S,
                                                       note_off_treatment=tok.spm_type_to_note_off_treatment(cs.SPM_TYPE),
                                                       mask_locations=masks,
                                                       measure_slice=(nn_input.start_measure, nn_input.end_measure + 1),
                                                       include_heads_for_empty_masked_measures=False,
                                                       track_measure_commands=user_commands.track_measure_commands,
                                                       explicit_rhythmic_conditioning_locations=user_commands.rhythmic_conditioning_locations,
                                                       rhythmic_conditioning_type=rhythmic_conditioning_type,
                                                       return_labels_too=False,
                                                       extra_id_st=extra_id_st,
                                                       extra_id_max=255,
                                                       commands_at_end=user_commands.commands_at_end,
                                                       )

        # has_fully_masked_inst parameter isn't quite correct this way, but it's ok
        if masks:
            labels = call_nn_infill(s, S=S, use_sampling=True, min_length=2, enc_no_repeat_ngram_size=0,
                                    has_fully_masked_inst=mask_pattern_type == 2, temperature=temperature)
        else:
            labels = ''  # save some time by not calling the NN if nothing is masked this cycle

        instructions_by_extra_id = nns.instructions_by_extra_id(labels)
        for eid, instruction_list in instructions_by_extra_id.items():
            if eid in extra_id_to_mask:
                write_to_S_for_variation(S=S, measure_i=extra_id_to_mask[eid][1], track_i=extra_id_to_mask[eid][0],
                                         instruction_list=instruction_list)

    # Make room for new S
    MEs = mt.get_measure_endpoints_in_QN(nn_input.start_measure, nn_input.end_measure)
    for T, take_w_info in track_measure_to_take_w_info.items():
        take = take_w_info.take if take_w_info else None
        if take:
            # qn_0 = mt.time_to_QN(time_selection[0])
            # qn_1 = mt.time_to_QN(time_selection[1])
            qn_0 = MEs[0]
            qn_1 = MEs[-1]
            eps = 0.0001
            notes = mm.get_notes(take=take, QN_window=[qn_0 - eps, qn_1 - eps])
            mm.delete_notes_by_idx(take=take, L=notes)

    # write new S to project
    cur_noteidx_rpr = 0
    written_takes = set()
    for track_i, track in enumerate(S.tracks):
        d_track = track.get_noteidx_info_dict(measure_lengths=S.get_measure_lengths())
        for measure_i, m_track in enumerate(track.tracks_by_measure):
            if (track_i, measure_i) in track_measure_to_take_w_info:
            # if nn_input.start_measure <= measure_i <= nn_input.end_measure:
                to_write = {}
                # take_w_info = find_take_w_info(measure_i, track.extra_info['reaper_track_group'], AT_by_track_idx)
                take_w_info = track_measure_to_take_w_info[(track_i, measure_i)]
                take = take_w_info.take if take_w_info else None
                measure_st_qn = nn_input.MEs_qn[measure_i]
                for n in m_track.note_ons:
                    length_in_clicks = d_track[n.noteidx].length
                    pitch = n.pitch
                    vel = n.vel
                    note_start_qn = measure_st_qn + (n.click / CPQ)
                    note_end_qn = note_start_qn + (length_in_clicks / CPQ)
                    startppq = mm.QN_to_ppq(x_in_QN=note_start_qn, take=take)
                    endppq = mm.QN_to_ppq(x_in_QN=note_end_qn, take=take)
                    note = mm.MyNote(take=take, noteidx=cur_noteidx_rpr, selected=new_notes_are_selected, muted=0,
                                     startppqpos=startppq, endppqpos=endppq, chan=0, pitch=pitch, vel=vel)
                    to_write[cur_noteidx_rpr] = note
                    cur_noteidx_rpr += 1
                mm.correct_note_overlaps(d=to_write)
                to_write = list(to_write.values())
                mm.insert_MyNotes(take=take, notes=to_write, noSortInOptional=True)
                written_takes.add(take)
    for take in written_takes:
        mm.MIDI_sort(take)

    return True


def write_to_S_for_variation(S, measure_i, track_i, instruction_list):
    """uses instruction list to overwrite (track_i, measure_i) in S"""
    d = {}
    for track in S.tracks:
        d_track = track.get_noteidx_info_dict()
        d.update(d_track)
    if d:
        m = max(d)
    else:
        m = -1
    st_note_id = m + 1

    T = S.tracks[track_i][measure_i]
    vel_positions = []
    vels = []
    idxs_to_remove = set()
    for n in T.note_ons:
        idxs_to_remove.add(n.noteidx)
        vel_positions.append(n.click)
        vels.append(n.vel)

    # remove from note_ons
    T.note_ons = [n for n in T.note_ons if n.noteidx not in idxs_to_remove]

    # remove from note_offs
    for bmt in S.tracks[track_i]:
        bmt.note_offs = [n for n in bmt.note_offs if n.noteidx not in idxs_to_remove]

    # add new note_ons and note_offs
    if tok.spm_type_to_note_off_treatment(cs.SPM_TYPE) not in ('duration', 'exclude'):
        raise NotImplemented('Writing instruction strings of type "duration" and "exclude" are'
                             ' the only types currently implemented')

    cur_note_id = st_note_id
    cur_pos = 0
    cur_dur = CPQ // 2  # default
    for instruction in instruction_list:
        if instruction[0] in ('w', 'd', 'N', 'D'):
            s = instruction.split(':')
            if s[0] == 'd':
                cur_dur = int(s[1])
            elif s[0] == 'w':
                cur_pos += int(s[1])
            else:
                pitch = int(s[1])
                vel_index = mf.index_of_closest_element_in_sorted_numeric_list(vel_positions, cur_pos)
                vel = vels[vel_index]
                note_on = ms.NoteOn(pitch=pitch, vel=vel, click=cur_pos, noteidx=cur_note_id)
                note_off = ms.NoteOff(pitch=pitch, click=cur_pos + cur_dur, noteidx=cur_note_id)
                T.note_ons.append(note_on)
                T.note_offs.append(note_off)
                cur_note_id += 1


def call_nn_infill(s, S, use_sampling=True, min_length=10, enc_no_repeat_ngram_size=0, has_fully_masked_inst=False,
                   temperature=1.0):
    from xmlrpc.client import ServerProxy
    import xmlrpc
    try:
        proxy = ServerProxy('http://127.0.0.1:3456')
        # cannot use keyword arguments, so make sure everything lines up
        res = proxy.call_nn_infill(s, pre.encode_midisongbymeasure_to_save_dict(S), use_sampling, min_length, enc_no_repeat_ngram_size, has_fully_masked_inst,
                                   temperature)
    except Exception as exception:
        if type(exception) == xmlrpc.client.Fault:
            print('Exception raised by NN:')
        else:
            errormsg = 'NN server not found. '
            errormsg += 'Make sure you have started the NN server manually by running composers_assistant_nn_server.py (or .exe).'
            mt.messagebox(msg=errormsg,
                          title='REAPER: NN server error',
                          int_type=0)
        raise exception

    return res
