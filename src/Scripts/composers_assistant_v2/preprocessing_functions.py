import constants
import midisong as ms
import collections
import copy


# test written
def apply_simplified_drum_map(S: "ms.MidiSong"):
    """in place operation"""
    for t in S.tracks:
        if t.is_drum:
            for n in t.notes:
                n.pitch = constants.SIMPLIFIED_DRUM_MAP[n.pitch]
            t.notes = ms.remove_duplicate_events_at_same_click(t.notes, cmp=lambda x: x.pitch)
            t.notes = [n for n in t.notes if n.pitch >= 0]


# test written
def _perform_all_possible_non_overlapping_consolidations(S: "ms.MidiSongByMeasure") -> "ms.MidiSongByMeasure":
    """consumes calling track.
    Make sure S already has drum track (if present) (inst==128) at the end before calling this function"""
    idx = S.find_one_pair_of_non_overlapping_track_consolidation_indexes()
    while idx is not None:
        # track order is maintained when switching back to MidiSong form, but drum track(s) get moved to the end.
        # So make sure S already has drum track at end before calling this function
        S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True)
        t1 = S.tracks.pop(idx[0])
        t2 = S.tracks.pop(idx[1] - 1)
        new_track = ms.combine_tracks(L=[t1, t2], inst_for_result=t1.inst)
        S.tracks.insert(idx[0], new_track)
        S = ms.MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
        idx = S.find_one_pair_of_non_overlapping_track_consolidation_indexes()

    return S


# done; helper function
def _equality_measure(tr1: ms.Track, tr2: ms.Track):
    import portion
    by_pitch_1 = collections.defaultdict(portion.Interval)
    by_pitch_2 = collections.defaultdict(portion.Interval)
    for n in tr1.notes:
        by_pitch_1[n.pitch] = by_pitch_1[n.pitch].union(portion.closed(n.click, n.end))
    for n in tr2.notes:
        by_pitch_2[n.pitch] = by_pitch_2[n.pitch].union(portion.closed(n.click, n.end))

    num = 0
    denom = 0

    # alt denom computations
    alt_denom_candidate_1 = 0
    alt_denom_candidate_2 = 0
    for pitch in by_pitch_1:
        for x in by_pitch_1[pitch]:
            if not x.empty:
                alt_denom_candidate_1 += x.upper - x.lower
    for pitch in by_pitch_2:
        for x in by_pitch_2[pitch]:
            if not x.empty:
                alt_denom_candidate_2 += x.upper - x.lower
    alt_denom = max(alt_denom_candidate_1, alt_denom_candidate_2)

    # numerator and old denominator computations
    for pitch in set(by_pitch_1.keys()).union(set(by_pitch_2.keys())):
        union = by_pitch_1[pitch].union(by_pitch_2[pitch])
        for x in union:
            if not x.empty:
                denom += x.upper - x.lower

        intersection = by_pitch_1[pitch].intersection(by_pitch_2[pitch])
        for x in intersection:
            if not x.empty:
                num += x.upper - x.lower

    if denom == 0.0:
        return 0
    else:
        return num/alt_denom  # what percentage of the larger one is accounted for by the smaller one


# helper function
def _get_first_note_of_pitch(pitch, L):
    for n in L:
        if n.pitch == pitch:
            return n


# helper function
def _get_note_shift_guess_1(tr1, tr2, cpq):
    res = 0
    if tr1.notes and tr2.notes:
        # find a common pitch
        pitches = set(n.pitch for n in tr1.notes)
        pitches = pitches.intersection(set(n.pitch for n in tr2.notes))
        pitches = sorted(list(pitches))
        if pitches:
            n = _get_first_note_of_pitch(pitch=pitches[-1], L=tr1.notes)
            n2 = _get_first_note_of_pitch(pitch=pitches[-1], L=tr2.notes)
            res = n.click - n2.click
    if abs(res) < 2 * cpq:  # shift at most one half note
        return res
    else:
        return 0


# helper function
def _get_note_shift_guess_2(tr1, tr2, cpq):
    res = 0
    if tr1.notes and tr2.notes:
        # get the first note from each
        res = tr1.notes[0].click - tr2.notes[0].click
    if abs(res) < 2 * cpq:  # shift at most one half note
        return res
    else:
        return 0


# test written
def _perform_all_near_equal_track_removals(S: "ms.MidiSong", threshold=1.0):
    """in place operation"""

    insts = collections.Counter()
    for t in S.tracks:
        insts[t.inst] += 1
    # insts is the set of instruments that appear in > 1 track
    insts = set(inst for inst in insts if insts[inst] > 1)

    removals = set()
    for inst in insts:
        tracks = [(i, t) for i, t in enumerate(S.tracks) if t.inst == inst]  # here are all the tracks for this inst
        for i, T1 in enumerate(tracks):
            for T2 in tracks[i+1:]:
                t1, t2 = T1[1], T2[1]
                if t1 not in removals:  # only compare tracks to t1 if t1 is not already slated for removal
                    g_1 = _get_note_shift_guess_1(t1, t2, S.cpq)
                    g_2 = _get_note_shift_guess_2(t1, t2, S.cpq)
                    shift_guesses = [0]
                    if g_1 not in shift_guesses:
                        shift_guesses.append(g_1)
                    if g_2 not in shift_guesses:
                        shift_guesses.append(g_2)
                    for shift_amt in shift_guesses:
                        t2_shifted = copy.copy(t2)
                        for n in t2_shifted.notes:
                            n.click += shift_amt
                            n.end += shift_amt
                        if _equality_measure(t1, t2_shifted) > threshold:
                            # print('ABOVE threshold:', t1.inst, shift_amt, _equality_measure(t1, t2_shifted))
                            removals.add(T2[0])
                            break

    removals = sorted(list(removals))
    for i, removal_index in enumerate(removals):
        S.tracks.pop(removal_index - i)


# Done. Note: Fairly slow!
def load_and_clean_midisongbymeasure_from_midi_path(p, quantize=None):
    if quantize is None:
        quantize = constants.QUANTIZE
    try:
        S = ms.MidiSong.from_midi_file(p)  # automatically puts drum track at the end
    except Exception as e:
        print(f'Exception {e} loading file: {p}. Skipping this file.')
        return None

    S.remove_tracks_with_no_notes()
    S.apply_pedals_to_extend_note_lengths()
    S.fix_note_overlaps()
    S.remove_ccs_and_pitch_bends()
    S.remove_pedals()
    apply_simplified_drum_map(S)

    # perform near-equal consolidations (throw out tracks that are near-identical) before quantizing
    # It is important to do this before consolidating non-overlapping tracks
    _perform_all_near_equal_track_removals(S, threshold=0.9)

    # note that preprocessed midis are NOT transposed into acceptable ranges yet.
    # transposition into acceptable ranges happens before sentencepiece training, and during augmentation during
    # train/val/testing.
    # transpose_into_acceptable_ranges_TT(S)

    # change to by-measure
    S = ms.MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
    S.quantize_notes_by_measure(q=quantize)  # also cleans up note duplicates at same click
    S.change_cpq(to_cpq=ms.extended_lcm(quantize))
    # changing cpq can cause tempo change collisions, so clean that up
    S.tempo_changes = ms.remove_duplicate_events_at_same_click(S.tempo_changes, cmp=lambda x: x.click)
    S.remove_empty_measures_at_beginning_and_end()
    S.remove_every_empty_measure_that_has_an_empty_preceding_measure()
    MEs = S.get_measure_endpoints()

    # Consolidate non-overlapping tracks now, using quantized measure coverage information.
    S = _perform_all_possible_non_overlapping_consolidations(S)

    # change back to full song
    S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True)

    S.sort_tracks_by_inst_and_avg_note_pitch()

    for t in S.tracks:
        t.sort()

    # back to MidiSongByMeasure one last time
    S = ms.MidiSongByMeasure.from_MidiSong(S, measure_endpoints=MEs, consume_calling_song=True)
    return S


# note: doesn't save cc's or pitch bends or markers etc.
# test written
def encode_midisongbymeasure_to_save_dict(S: "ms.MidiSongByMeasure"):
    res = {}
    res['cpq'] = S.cpq
    res['MEs'] = S.get_measure_endpoints()
    res['track_insts'] = [t.inst for t in S.tracks]
    res['tempo_changes'] = [(t.val, t.click) for t in S.tempo_changes]
    res['tracks'] = []
    for t in S.tracks:
        this_track_info = []
        for m_t in t.tracks_by_measure:
            this_measure = []
            for n in m_t.note_ons:
                this_measure.append('{};{};{};{}'.format(n.pitch, n.click, n.noteidx, n.vel))  # N C X V format
            note_ons = ' '.join(this_measure)
            this_measure = []
            for n in m_t.note_offs:
                this_measure.append('{};{}'.format(n.click, n.noteidx))  # C X format
            note_offs = ' '.join(this_measure)
            m_t_info = (note_ons, note_offs)
            this_track_info.append(m_t_info)
        res['tracks'].append(this_track_info)
    return res


# test written
def midisongbymeasure_from_save_dict(d: dict) -> ms.MidiSongByMeasure:
    def proc_notes(n_ons, n_offs, idx_d):
        """alters idx_d in place"""
        note_ons_res = []
        if n_ons:  # if n_ons is not an empty string
            for s in n_ons.split(' '):
                N, c, X, V = s.split(';')
                N = int(N)
                c = int(c)
                X = int(X)
                V = int(V)
                note_ons_res.append(ms.NoteOn(pitch=N, click=c, noteidx=X, vel=V))
                idx_d[X] = N  # idx to pitch
        note_offs_res = []
        if n_offs:
            for s in n_offs.split(' '):  # if n_offs is not an empty string
                c, X = s.split(';')
                c = int(c)
                X = int(X)
                note_offs_res.append(ms.NoteOff(pitch=idx_d[X], click=c, noteidx=X))
        return note_ons_res, note_offs_res

    tracks = []

    for t_i, t in enumerate(d['tracks']):
        inst = d['track_insts'][t_i]
        tracks_by_measure = []
        idx_tracker = {}
        for t_m in t:
            note_ons, note_offs = t_m
            note_ons, note_offs = proc_notes(note_ons, note_offs, idx_tracker)
            this_track = ms.Track(inst=inst,
                                  note_ons=note_ons,
                                  note_offs=note_offs)
            tracks_by_measure.append(this_track)
        tracks.append(ms.ByMeasureTrack(inst=inst, tracks_by_measure=tracks_by_measure))

    tempo_changes = [ms.TempoChange(val=t[0], click=t[1]) for t in d['tempo_changes']]
    return ms.MidiSongByMeasure(tracks=tracks, measure_endpoints=d['MEs'], tempo_changes=tempo_changes, cpq=d['cpq'])


# for parallel processing purposes
def preprocess_midi_to_save_dict(p, quantize=None):
    """p a path"""
    S = load_and_clean_midisongbymeasure_from_midi_path(p, quantize=quantize)
    if S is not None:
        d = encode_midisongbymeasure_to_save_dict(S)
    else:
        d = None
    return p, d
