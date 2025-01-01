from midisong import *
import numpy as np


def cosine_similarity(v1, v2):
    """v1, v2 1-d numpy arrays of equal length"""
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ex:
# v1 = (1,1,1,1,1,1,1,1,1,1,1,1)
# v = (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)
# cosine_similarity(v1, v)


def arccos(theta):
    if theta >= 1.0:
        return 0
    return np.arccos(theta)


def note_onset_counts_mod_qn(S: MidiSongByMeasure):
    """S a MidiSongByMeasure"""
    res = np.zeros(S.cpq)
    for t in S.tracks:
        for m in t.tracks_by_measure:
            for n in m.note_ons:
                res[n.click % S.cpq] += 1
    return res


def fix_up(p):  # unused, just here for future use elsewhere
    S = MidiSong.from_midi_file(p)
    S.remove_tracks_with_no_notes()
    S.apply_pedals_to_extend_note_lengths()
    S.fix_note_overlaps()
    S.remove_ccs_and_pitch_bends()
    S.remove_pedals()

    # change to by-measure
    S = MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
    S.quantize_notes_by_measure(q=(4, 3))
    S.change_cpq(12)
    S.remove_empty_measures_at_beginning_and_end()
    S.remove_every_empty_measure_that_has_an_empty_preceding_measure()

    # change back to full song
    S = MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True)
    return S


def get_deduping_and_filtering_info(p):
    res = {}
    res['p'] = p
    res['error'] = False

    try:
        S = MidiSong.from_midi_file(p)
    except:
        res['error'] = True
        res['cos_sim'] = 1.0
        res['dedupe_strs'] = []
        return res

    S.remove_tracks_with_no_notes()
    # remove drum track
    new_tracks = [t for t in S.tracks if not t.is_drum]

    S.tracks = new_tracks
    S.apply_pedals_to_extend_note_lengths()
    S.fix_note_overlaps()
    S.remove_ccs_and_pitch_bends()
    S.remove_pedals()

    # change to by-measure
    try:
        S = MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
    except:
        res['error'] = True
        res['cos_sim'] = 1.0
        res['dedupe_strs'] = []
        return res

    S_unquantized = copy.copy(S)
    S.quantize_notes_by_measure(q=(4, 3))
    S.change_cpq(12)
    S.remove_empty_measures_at_beginning_and_end()
    S.remove_every_empty_measure_that_has_an_empty_preceding_measure()
    res['n_measures'] = S.get_n_measures()

    # compute cos sim
    S_unquantized.change_cpq(12)
    v1 = note_onset_counts_mod_qn(S_unquantized)
    v2 = np.ones(S_unquantized.cpq)
    if sum(v1) > 0:  # if we have notes
        cos_sim = cosine_similarity(v1, v2)
    else:
        cos_sim = 1.0
        print('no notes; n_measures = {}'.format(res['n_measures']))
    res['cos_sim'] = cos_sim

    # Switch back to not-by-measure
    S = MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True)

    # compute deduping strings
    dedup_strs = [S.to_deduping_str()]
    for k in range(11):
        S.transpose(1)
        dedup_strs.append(S.to_deduping_str())
    res['dedupe_strs'] = dedup_strs

    return res
