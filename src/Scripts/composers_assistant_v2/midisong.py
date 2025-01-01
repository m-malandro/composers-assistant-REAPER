# TODO: For future projects, transpose into acceptable range is the functionality that
#       needs to be fixed to keep longest notes.
#       The combine_tracks function in this file keeps longest notes.

# CONVENTIONS:
# -instrument 128 = drums
# -A track contains notes, ccs, etc., usually for a single instrument. Instrument information is typically stored at
#  the track level, but it can also be stored in each note, cc, etc. if the track contains multiple instruments. If
#  the track contains multiple instruments (potentially), that is signified by setting track.inst = None
# -A MidiSong is conceptually a list of tracks together with some additional information.
# -Almost everything is mutable, and almost all operations occur in place. Function documentation always tells you
#  when something happens in place.

############
# miditoolkit usage notes:

# M = miditoolkit.Midifile(filename)

# M has important properties
# .ticks_per_beat
# .max_tick
# .time_signature_changes
# .tempo_changes
# .instruments

# M.instruments is a list of tracks
# M.instruments[i] has attributes:
# .notes (a list)
# .is_drum (bool)
# .program (int)
# .name (str, basically useless)
# .control_changes (list)
# .pedals (list)
# .pitch_bends (list)
# Q: What about poly pressure (PoPr) and channel pressure (ChPr)?
# A: These appear not to be recorded.
# Proof: See LMD files:
# 000c856455f06e47924c6948966163db.mid (has PoPr), and
# 000c923c18c62332503050809ea3eb8f.mid (has ChPr).

# M.instruments[i].notes[k] is a Note object (theirs), with attributes:
# .start, .end, .pitch, .velocity (all ints; start and end are clicks)
# == appears not to be implemented for their Note's.

# M.instruments[i].control_changes[k] is a ControlChange object (theirs), with attributes:
# .number, .value, .time.
# Note that cc64's are in this list alongside the Pedal objects they create (see below)

# M.instruments[i].pedals[k] is a Pedal object (theirs), with attributes:
# .start, .end

# M.instruments[i].pitch_bends[k] is a PitchBend object (theirs), with attributes:
# .pitch, .time

# note that miditoolkit.dump does not respect channels (except channel == 9 for drum tracks); rather, it just cycles
# through all the non-drum channels for each non-drum track it gets, using the convention that each of its
# tracks contains a single instrument.
#############

import collections
import copy
import itertools
import bisect
import math
import typing
import statistics

from containers import *

_WARNINGS_PRINTED = set()


##################################################
# Primary classes

# Basic Track class.
# Inspired by miditoolkit.Instrument, but can handle multiple instruments
# (since each note in self.notes can have an .inst value).
# Can also hold NoteOn's and NoteOff's
class Track(object):
    @property
    def inst(self):
        return self._inst

    @inst.setter
    def inst(self, x):
        self._inst = x
        self.is_drum = (self.inst == 128)

    def __init__(self, inst=None, notes=None, note_ons=None, note_offs=None,
                 ccs=None, pitch_bends=None, pedals=None, name='', extra_info=None):
        """Basic class for handling "tracks". Events within e.g. .notes cannot be assumed to be sorted in any way,
        unless you call one of the self.sort methods or do a sort of its attributes on your own first.
        This class will never sort anything for any other reason.

        use inst=None if you want the instrument represented by your Track instance to be
        controlled at the measure level or note level.

        :type notes: list[Note]
        :type note_ons: list[NoteOn]
        :type note_offs: list[NoteOff]
        :type ccs: list[CC]
        :type pitch_bends: list[PitchBend]
        :type pedals: list[Pedal]
        :type extra_info: dict
        """
        self.inst = inst

        self.notes = notes if notes is not None else []
        self.note_ons = note_ons if note_ons is not None else []
        self.note_offs = note_offs if note_offs is not None else []

        self.ccs = ccs if ccs is not None else []
        self.pedals = pedals if pedals is not None else []
        self.pitch_bends = pitch_bends if pitch_bends is not None else []
        self.extra_info = extra_info if extra_info is not None else {}

        self.name = name

    @classmethod
    def from_miditoolkit_Instrument(cls, X):
        if X.is_drum:
            inst = 128
        else:
            inst = int(X.program)

        notes = [Note.from_miditoolkit_Note(note) for note in X.notes]
        ccs = [CC.from_miditoolkit_ControlChange(cc) for cc in X.control_changes]
        pitch_bends = [PitchBend.from_miditoolkit_PitchBend(p) for p in X.pitch_bends]
        pedals = [Pedal.from_miditoolkit_Pedal(p) for p in X.pedals]

        name = X.name

        return cls(inst=inst, notes=notes, ccs=ccs, pitch_bends=pitch_bends, pedals=pedals, name=name)

    # test written
    @classmethod
    def from_ByMeasureTrack(cls, track: 'ByMeasureTrack', measure_endpoints: list, consume_calling_track=False):
        """puts the measures from calling track (type ByMeasureTrack) back together into a single Track,
        as well as possible. If consume_calling_track, then calling track is essentially destroyed,
        and should not be used again.

        Uses the calling track's .inst as the .inst of the result.
        """
        if not consume_calling_track:
            track = copy.copy(track)
        calling_track = track

        res = Track(inst=calling_track.inst, name=calling_track.name, extra_info=calling_track.extra_info)
        measure_st_clicks = measure_endpoints[:-1]
        for name in res.all_names_of_iterables_with_time_events():
            for measure_i, measure_st_click in enumerate(measure_st_clicks):
                L = getattr(calling_track.tracks_by_measure[measure_i], name)
                for evt in L:
                    evt.click += measure_st_click
                getattr(res, name).extend(L)

        # TODO impute missing note offs (if any) before joining
        res.join_note_ons_and_note_offs_to_notes()

        return res

    def has_notes(self):
        return bool(self.notes) or bool(self.note_ons)

    def sort_notes(self):
        """in place operation"""
        # sorts are stable
        self.notes.sort(key=lambda x: x.end)
        self.notes.sort(key=lambda x: x.vel)
        self.notes.sort(key=lambda x: x.pitch)
        self.notes.sort(key=lambda x: x.click)

    def sort_note_ons(self):
        """in place operation"""
        self.note_ons.sort(key=lambda x: x.vel)
        self.note_ons.sort(key=lambda x: x.pitch)
        self.note_ons.sort(key=lambda x: x.click)

    def sort_note_offs(self):
        """in place operation"""
        self.note_offs.sort(key=lambda x: x.pitch)
        self.note_offs.sort(key=lambda x: x.click)

    def sort_ccs(self):
        """in place operation"""
        self.ccs.sort(key=lambda x: x.val)
        self.ccs.sort(key=lambda x: x.cc)
        self.ccs.sort(key=lambda x: x.click)

    def sort_pitch_bends(self):
        """in place operation"""
        self.pitch_bends.sort(key=lambda x: x.val)
        self.pitch_bends.sort(key=lambda x: x.click)

    def sort_pedals(self):
        """in place operation"""
        self.pedals.sort(key=lambda x: x.end)
        self.pedals.sort(key=lambda x: x.click)

    def sort(self):
        """In place operation. self.notes will be ordered by click, then at a given click by pitch,
        then by velocity, then by end. note.inst is irrelevant for this sorting.

        ccs, pedals, and pitch_bends are sorted primarily by click, too."""
        self.sort_notes()
        self.sort_note_ons()
        self.sort_note_offs()
        self.sort_ccs()
        self.sort_pitch_bends()
        self.sort_pedals()

    def __repr__(self):
        res = ["Tr inst:{}".format(self.inst)]
        events = []
        events.extend(self.notes)
        events.extend(self.note_ons)
        events.extend(self.note_offs)
        events.extend(self.ccs)
        events.extend(self.pedals)
        events.extend(self.pitch_bends)
        # more extensive sorting is required if we want to use __repr__ to test equality
        events.sort(key=lambda x: x.click)
        res.extend([str(x) for x in events])

        return '(' + ' '.join(res) + ')'

    @abstractmethod
    def all_names_of_iterables_with_time_events(self) -> list:
        return ['notes', 'note_ons', 'note_offs', 'ccs', 'pedals', 'pitch_bends']

    @abstractmethod
    def all_iterables_with_time_events(self) -> list:
        """must return a list L of lists K, so that other functions can iterate over the elements of each K"""
        res = []
        for name in self.all_names_of_iterables_with_time_events():
            to_append = getattr(self, name)
            if to_append:
                res.append(to_append)
        return res

    # test written
    def get_events_by_measure(self, measure_endpoints, which_events=('notes',)) -> dict:
        """which_events a tuple of strings, each of which is an attribute of the calling track, or "all", which gets
        all types of events.

        measure_endpoints must be a sorted list of integers, starting with 0, containing no repeated values.

        Returns a dict d whose keys are the elements of which_events.

        d[key] will be a list of length len(measure_endpoints)-1, so a list of length equal to the number of
        measures. Each element L of this list is itself a list of events of key type in the corresponding measure.

        If the elements of self.key are sorted by click, then so will each list L in d[key].
        """
        if which_events == "all":
            which_events = self.all_names_of_iterables_with_time_events()

        res = {}
        for event_type in which_events:
            events_by_measure = [[] for _ in range(len(measure_endpoints) - 1)]
            L = getattr(self, event_type)
            for evt in L:
                measure_idx = bisect.bisect_right(measure_endpoints, evt.click) - 1
                events_by_measure[measure_idx].append(evt)

            res[event_type] = events_by_measure
        return res

    # test written
    def quantize_notes(self, cpq, q=(4, 3), quantize_end_too=True):
        """Alters self.notes in place by altering their .click attributes. This may alter their .end attributes too,
        just by the nature of moving .click's to the right.

        If quantize_end_too, then note.end attributes are explicitly quantized as well.

        This function only looks at self.notes. self.note_ons and self.note_offs are untouched.

        You probably *don't* actually want to use this function. It is much more likely that you want to convert
        this Track into a ByMeasureTrack (using ByMeasureTrack.from_track), and then run
        quantize_notes_by_measure on that ByMeasureTrack.
        """
        if quantize_end_too:
            note_ons, note_offs = compute_split_notes_to_note_ons_and_note_offs(self.notes)
            quantize_list(L=note_ons, start_click=0, end_click=None, cpq=cpq, q=q)
            quantize_list(L=note_offs, start_click=0, end_click=None, cpq=cpq, q=q)
            # then put this back together into Notes
            notes = compute_join_note_ons_and_note_offs_to_notes(note_ons, note_offs)
            self.notes = notes

        else:
            # quantize all of self.notes
            quantize_list(L=self.notes, start_click=0, end_click=None, cpq=cpq, q=q)

        # if two notes quantize to the same start position, clean that up.
        self.notes = remove_duplicate_events_at_same_click(self.notes, lambda x: x.pitch)

    # test written
    def split_notes_to_note_ons_and_note_offs(self):
        """resets self.notes to [] after setting self.note_ons and self.note_offs to the corresponding NoteOn + NoteOff
        information. Such information is placed into self.note_ons and self.note_offs in the order it was
        found in self.notes. NoteOn and NoteOff events have their noteidx properties set as well.

        For safety purposes, has no effect if self.notes is empty"""
        if self.notes:
            self.note_ons, self.note_offs = compute_split_notes_to_note_ons_and_note_offs(self.notes)
            self.notes = []

    # test written
    def join_note_ons_and_note_offs_to_notes(self):
        """resets self.note_ons and self.note_offs to [] after setting self.notes to the information obtained from them.
        For this to work, elements of self.note_ons and self.note_offs must have their .noteidx attributes set.

        For safety purposes, has no effect if self.note_ons is empty or self._note_offs is empty"""
        if self.note_ons and self.note_offs:
            self.notes = compute_join_note_ons_and_note_offs_to_notes(note_ons=self.note_ons, note_offs=self.note_offs)
            self.note_ons, self.note_offs = [], []

    # small test written
    def __copy__(self):
        d = {}
        for name in self.all_names_of_iterables_with_time_events():
            L = getattr(self, name)
            d[name] = [copy.copy(x) for x in L]
        extra_info = copy.deepcopy(self.extra_info)
        return type(self)(inst=self.inst, name=self.name, extra_info=extra_info, **d)

    def transpose(self, amt):
        """in-place operation. Note: This will transpose the track even if it's a drum track."""
        for L in [self.notes, self.note_ons, self.note_offs]:
            transpose_list(L, amt=amt)

    def is_empty(self):
        for iterable in self.all_iterables_with_time_events():
            if iterable:
                return False
        return True

    # test written
    def transpose_by_octaves_into_range(self, range_=(0, 127)):
        """in place operation. Transposes all .pitches by 12 at a time until they are such that
        range_[0] <= pitch <= range[1]. It is recommended to use the function remove_duplicate_events_at_same_click
        or otherwise dedupe after calling this function to clean up possible collisions caused by this function."""
        if range_[1] - range_[0] < 12:
            raise ValueError('range_ must be at least one octave')
        low, hi = range_
        for L in [self.notes, self.note_ons, self.note_offs]:
            for n in L:
                while n.pitch < low:
                    n.pitch += 12
                while n.pitch > hi:
                    n.pitch -= 12

    # test written
    def is_poly(self):
        """returns True iff there is a click c such that there are two note onsets at c. Looks at self.note_ons and
        self.notes"""
        clicks_seen = set()
        for n in itertools.chain(self.notes, self.note_ons):
            c = n.click
            if c in clicks_seen:
                return True
            clicks_seen.add(c)
        return False


class ByMeasureTrack(object):
    @property
    def inst(self):
        return self._inst

    @inst.setter
    def inst(self, x):
        self._inst = x
        self.is_drum = (self.inst == 128)

    def __init__(self, inst=None, tracks_by_measure=None, name="", extra_info=None):
        """
        tracks_by_measure a list of Tracks containing nothing that has an .end attribute.
        This list has length equal to the number of measures represented by this object.

        Note that the .click's of things in tracks_by_measure are understood to be RELATIVE to the start of the
        corresponding measure. The measure endpoint clicks are not stored in this object itself - usually they are
        stored in a MidiSongByMeasure that contains ByMeasureTrack's in its .tracks.

        Use inst=None if you want the instrument represented by your ByMeasureTrack instance to be
        controlled at the measure level or note level.

        :type inst: int or None
        :type tracks_by_measure: list[Track]
        :type name: str
        :type extra_info: dict
        """

        self.inst = inst

        for t in tracks_by_measure:
            if t.pedals:
                raise ValueError('ByMeasureTrack does not support pedals')

        self.tracks_by_measure = tracks_by_measure if tracks_by_measure is not None else []
        self.extra_info = extra_info if extra_info is not None else {}

        self.name = name

    # test written
    @classmethod
    def from_Track(cls, track: Track, measure_endpoints: list, consume_calling_track=False):
        """If consume_calling_track, then you should never use the input track again in any way, as
        it gets altered and its attributes get bound to the output of this function.
        Use consume_calling_track=False if you want the input track to be unaffected by this function.

        Uses calling track's .inst as the .inst of both the resulting ByMeasureTrack, as well as the .inst of every
        Track contained in it.

        :type track: Track
        :type measure_endpoints: list[int]
        :type consume_calling_track: bool"""

        n_measures = len(measure_endpoints) - 1

        if not consume_calling_track:
            track = copy.copy(track)  # so as not to screw up the original track

        track.split_notes_to_note_ons_and_note_offs()

        EBM = track.get_events_by_measure(measure_endpoints=measure_endpoints,
                                          which_events="all")

        # ex: EBM['notes'] = [ [measure 1 notes], [measure 2 notes], etc.]
        # we are essentially transposing that info here
        tracks_by_measure = []
        for measure_idx in range(n_measures):
            d = {}
            for k, L in EBM.items():
                d[k] = L[measure_idx]
            new_track = Track(inst=track.inst, **d)
            tracks_by_measure.append(new_track)

        for i, new_track in enumerate(tracks_by_measure):
            measure_start_click = measure_endpoints[i]
            for L in new_track.all_iterables_with_time_events():
                for event in L:
                    event.click -= measure_start_click
                    if hasattr(event, 'end'):
                        raise ValueError(f'ByMeasureTrack does not support events with "end" events, such as {event}')

        return ByMeasureTrack(inst=track.inst,
                              tracks_by_measure=tracks_by_measure,
                              name=track.name,
                              extra_info=track.extra_info)

    def __getitem__(self, item):
        return self.tracks_by_measure[item]

    def __iter__(self):
        for t in self.tracks_by_measure:
            yield t

    def __copy__(self):
        t_res = [copy.copy(x) for x in self.tracks_by_measure]
        extra_info = copy.deepcopy(self.extra_info)
        return ByMeasureTrack(inst=self.inst, tracks_by_measure=t_res, name=self.name, extra_info=extra_info)

    # test written
    def has_notes(self):
        for t in self.tracks_by_measure:
            if t.has_notes():
                return True
        return False

    def sort(self):
        """in place operation"""
        for t in self.tracks_by_measure:
            t.sort()

    def __repr__(self):
        res = ['ByMeasureTrack inst:{}'.format(self.inst)]
        for i, t in enumerate(self.tracks_by_measure):
            res.append('measure {}'.format(i))
            res.append(repr(t))
        return ' '.join(res)

    def all_iterables_with_time_events(self) -> list:
        """must return a list L of lists K, so that other functions can iterate over the elements of each list K"""
        res = []
        for t in self.tracks_by_measure:
            for iterable in t.all_iterables_with_time_events():
                res.append(iterable)
        return res

    # test written
    def quantize_notes_by_measure(self, cpq, measure_endpoints, q=(4, 3)):
        """Alters track.note_ons and track.note_offs in place for each track in self.tracks_by_measure,
        by altering the .click attributes of the NoteOn's and NoteOff's inside.

        May also move events from the end of one measure to the start of the next, if that's where they quantize to.

        If anything in the last measure of self.tracks quantizes to the start of the next (nonexistent when this
        function was called) track, then this function will add a measure to self.tracks accordingly. If that
        happens, then this function returns True. Otherwise, it returns False.
        """

        assert len(measure_endpoints) - 1 == len(self.tracks_by_measure)

        d_appended_measure = {}
        res = False

        for target_list in ('note_ons', 'note_offs'):
            to_st_of_next_measure = []
            for measure_idx, t in enumerate(self.tracks_by_measure):
                L = getattr(t, target_list)
                measure_length = measure_endpoints[measure_idx + 1] - measure_endpoints[measure_idx]
                for elt in reversed(to_st_of_next_measure):
                    elt.click = 0
                    L.insert(0, elt)
                idx_to_st_of_next_measure = quantize_list(L=L, start_click=0, end_click=measure_length,
                                                          cpq=cpq, q=q)
                to_st_of_next_measure = []
                for j, elt_idx in enumerate(idx_to_st_of_next_measure):
                    to_st_of_next_measure.append(L.pop(elt_idx - j))

            # if something gets quantized to the end of the last measure, then plan to create new measure
            if to_st_of_next_measure:
                for elt in to_st_of_next_measure:
                    elt.click = 0
                d_appended_measure[target_list] = [elt for elt in to_st_of_next_measure]

        if d_appended_measure:
            res = True
            new_track = Track(inst=self.tracks_by_measure[-1].inst, **d_appended_measure)
            self.tracks_by_measure.append(new_track)

        # If two NoteOn's quantize to the same start position, clean that up.
        self._cleanup_note_duplicates()

        return res

    def transpose(self, amt):
        """in place operation. Note: This will transpose the ByMeasureTrack even if it's a drum track."""
        for t in self.tracks_by_measure:
            t.transpose(amt=amt)

    # test written
    def change_cpq(self, from_cpq, to_cpq):
        """in place operation"""
        factor = to_cpq / from_cpq
        for iterable in self.all_iterables_with_time_events():
            for n in iterable:
                n.click = round(n.click * factor)

    # test written; more extensive testing is done when testing MidiSongByMeasure
    def remove_measures_by_index(self, L):
        """L a list of integers (measure indexes). In-place operation."""
        for i, ct in enumerate(L):
            self.tracks_by_measure.pop(ct - i)  # remove track i

    # test written
    def get_noteidx_info_dict(self, measure_lengths=None) -> "dict[int, NoteIdxInfo]":
        """Every NoteOff must have exactly one corresponding NoteOn. NoteOn's do not need a corresponding NoteOff.
        If measure_lengths is provided (as a list), then the resulting NoteIdxInfo's will have .length properties."""
        res = {}
        for i, t in enumerate(self.tracks_by_measure):
            for n in t.note_ons:
                if n.noteidx in res:
                    raise RuntimeError("two NoteOn's share the same noteidx")
                res[n.noteidx] = NoteIdxInfo(note_on=n, measure_note_on=i)
        for i, t in enumerate(self.tracks_by_measure):
            for n in t.note_offs:
                if n.noteidx not in res:
                    raise RuntimeError("NoteOff without corresponding NoteOn")
                if res[n.noteidx].measure_note_off is not None:
                    raise RuntimeError("two NoteOffs share the same noteidx")
                res[n.noteidx].note_off = n
                res[n.noteidx].measure_note_off = i

        if measure_lengths is not None:
            for idx, n in res.items():
                if n.note_off is not None:
                    n.length = sum(measure_lengths[n.measure_note_on: n.measure_note_off]) - n.note_on.click + n.note_off.click

        return res

    # test written
    def get_note_on_note_off_measure_coverage(self):
        res = [0 for _ in range(len(self.tracks_by_measure))]
        note_idx_dict = self.get_noteidx_info_dict()
        for note_idx, info in note_idx_dict.items():
            low = info.measure_note_on
            hi = info.measure_note_off
            if info.note_off.click == 0:
                hi -= 1
            if hi < low:
                hi = low
            for i in range(low, hi + 1):
                res[i] = 1
        return res

    def _cleanup_note_duplicates(self):
        note_idxs_removed = set()

        # first remove note on's with the same pitch at the same click
        for t in self.tracks_by_measure:
            new_note_ons = remove_duplicate_events_at_same_click(t.note_ons, cmp=lambda x: x.pitch)
            new_note_on_idxs = set(n.noteidx for n in new_note_ons)
            old_note_on_idxs = set(n.noteidx for n in t.note_ons)
            note_idxs_removed.update(old_note_on_idxs.difference(new_note_on_idxs))
            t.note_ons = new_note_ons

        # then remove the corresponding note_offs
        for t in self.tracks_by_measure:
            to_pop = []
            for i, n in enumerate(t.note_offs):
                if n.noteidx in note_idxs_removed:
                    to_pop.append(i)
            for i, idx in enumerate(to_pop):
                t.note_offs.pop(idx - i)

    # test written
    def transpose_by_octaves_into_range(self, range_=(0, 127), cleanup_note_duplicates=True):
        """in place operation"""
        for t in self.tracks_by_measure:
            t.transpose_by_octaves_into_range(range_=range_)
        if cleanup_note_duplicates:
            self._cleanup_note_duplicates()


# Basic Song class. Similar to miditoolkit.MidiFile, but does not import numpy
# Guaranteed to have a time sig event at click 0
# Guaranteed to have a tempo change event at click 0
# Note that unlike time_signatures, a MidiSong S may have TempoChange's with equal values
# next to each other in S.tempo_changes.
class MidiSong(object):
    @classmethod
    def from_midi_file(cls, path_to_file, clean_up_time_signatures=True):
        import miditoolkit
        M = miditoolkit.MidiFile(filename=path_to_file, clip=True)
        tracks = [Track.from_miditoolkit_Instrument(t) for t in M.instruments]
        for t in tracks:
            t.sort()
        time_signatures = [TimeSig.from_miditoolkit_TimeSignature(t) for t in M.time_signature_changes]
        if not time_signatures:
            time_signatures = [TimeSig(num=4, denom=4, click=0)]
        cpq = M.ticks_per_beat
        max_tick = M.max_tick
        markers = [Marker.from_miditoolkit_Marker(x) for x in M.markers]
        tempo_changes = [TempoChange.from_miditoolkit_TempoChange(x) for x in M.tempo_changes]
        return cls(tracks=tracks, time_signatures=time_signatures, markers=markers, cpq=cpq,
                   max_click_on_init=max_tick, tempo_changes=tempo_changes,
                   clean_up_time_signatures=clean_up_time_signatures)

    @classmethod
    def from_MidiSongByMeasure(cls, song: "MidiSongByMeasure", consume_calling_song=False,
                               clean_up_time_signatures=True):
        """If consume_calling_song, then you should never reuse the song input to this function. Use
        consume_calling_song = False to ensure that the calling song is not altered by this function."""
        if not consume_calling_song:
            song = copy.copy(song)

        tracks = [Track.from_ByMeasureTrack(track=track, consume_calling_track=consume_calling_song,
                                            measure_endpoints=song.get_measure_endpoints(make_copy=False))
                  for track in song.tracks]
        time_signatures = compute_time_signatures_from_measure_endpoints(
            measure_endpoints=song.get_measure_endpoints(make_copy=False), cpq=song.cpq)
        markers = song.markers
        cpq = song.cpq
        tempo_changes = song.tempo_changes
        return cls(tracks=tracks, time_signatures=time_signatures, markers=markers, cpq=cpq,
                   tempo_changes=tempo_changes, clean_up_time_signatures=clean_up_time_signatures)

    def __init__(self, tracks=None, time_signatures=None, markers=None, cpq=960, max_click_on_init=None,
                 tempo_changes=None, clean_up_time_signatures=True):
        """
        :type tracks: list[Track]
        :type time_signatures: list[TimeSig]
        :type markers: list[Marker]
        :type tempo_changes: list[TempoChange]
        """

        # handle tracks
        self.tracks = tracks if tracks is not None else []
        # always consolidate drum tracks into one track
        drum_tracks = []
        non_drum_tracks = []
        for t in self.tracks:
            if t.is_drum:
                drum_tracks.append(t)
            else:
                non_drum_tracks.append(t)
        drum_track = combine_tracks(drum_tracks, inst_for_result=128)
        self.tracks = []
        for track in non_drum_tracks:
            self.tracks.append(track)
        if drum_track.has_notes():
            self.tracks.append(drum_track)

        self.cpq = cpq

        self.markers = markers if markers is not None else []
        self.markers = remove_duplicate_events_at_same_click(self.markers, cmp=lambda x: x.text)

        self.tempo_changes = tempo_changes if tempo_changes is not None else []
        self.__fix_tempo_changes()

        self.max_click_on_init = max_click_on_init if max_click_on_init is not None else self.get_max_click()

        self.time_signatures = time_signatures if time_signatures is not None else [TimeSig(num=4, denom=4, click=0)]
        self.__init_cleanup_time_sigs(clean_up_time_signatures=clean_up_time_signatures)

    # noinspection DuplicatedCode
    def __fix_tempo_changes(self):
        self.tempo_changes.sort(key=lambda x: x.click)
        self.tempo_changes = remove_duplicate_events_at_same_click(self.tempo_changes, cmp=lambda x: x.click)
        if not self.tempo_changes:
            self.tempo_changes = [TempoChange(val=120, click=0)]
        if self.tempo_changes[0].click > 0:
            self.tempo_changes.insert(0, TempoChange(val=120, click=0))

    # test written
    def sort_tracks_by_inst(self):
        """in place operation"""
        self.tracks.sort(key=lambda t: t.inst if t.inst is not None else -1000)  # "None" insts go first

    # test written
    def sort_tracks_by_inst_and_avg_note_pitch(self):
        """in place operation. Looks only at notes, not note_ons."""
        def sort_key(tr):
            p = average_pitch(tr.notes)
            return -1 * round(p, 2)

        self.tracks = _track_sorter_by_inst_and_avg_note_pitch(tracks=self.tracks, sort_key_fn=sort_key)

    def __copy__(self):
        tracks = [copy.copy(x) for x in self.tracks]
        time_sigs = [copy.copy(x) for x in self.time_signatures]
        markers = [copy.copy(x) for x in self.markers]
        tempo_changes = [copy.copy(x) for x in self.tempo_changes]
        return type(self)(tracks=tracks, time_signatures=time_sigs, markers=markers, tempo_changes=tempo_changes,
                          cpq=self.cpq, max_click_on_init=self.max_click_on_init)

    @abstractmethod
    def all_iterables_with_time_events(self):
        """must return a list of lists"""
        res = []
        res.append(self.time_signatures)
        res.append(self.markers)
        res.append(self.tempo_changes)
        for t in self.tracks:
            for iterable in t.all_iterables_with_time_events():
                res.append(iterable)
        return res

    def __init_cleanup_time_sigs(self, clean_up_time_signatures=True):
        """in place operation"""
        # Ensure we have at least one time sig
        if not self.time_signatures:
            self.time_signatures = [TimeSig(num=4, denom=4, click=0)]

        self.time_signatures.sort(key=lambda x: x.click)

        # Ensure we have a time sig at time 0
        if self.time_signatures[0].click > 0:
            self.time_signatures.insert(0, TimeSig(num=4, denom=4, click=0))

        # maybe slow, but necessary
        if clean_up_time_signatures:
            self.time_signatures = clean_up_time_sigs(time_signatures=self.time_signatures, cpq=self.cpq,
                                                      final_click=self.max_click_on_init)

    def apply_pedals_to_extend_note_lengths(self):
        """in place operation. It is highly recommended to call self.fix_note_overlaps() after calling this function.

        only touches t.notes for t in self.tracks. t.note_ons and t.note_offs are not affected."""
        for t in self.tracks:
            apply_pedals_to_extend_note_lengths(t.notes, t.pedals)

    def fix_note_overlaps(self):
        """in place operation.

        only touches t.notes for t in self.tracks. t.note_ons and t.note_offs are not affected."""
        for t in self.tracks:
            fix_note_overlaps(t.notes)

    def remove_ccs_and_pitch_bends(self):
        """in place operation"""
        for t in self.tracks:
            t.ccs = []
            t.pitch_bends = []

    def remove_pedals(self):
        """in place operation"""
        for t in self.tracks:
            t.pedals = []

    # test written
    def remove_tracks_with_no_notes(self):
        """in place operation"""
        tracks = []
        for t in self.tracks:
            if t.has_notes():
                tracks.append(t)
        self.tracks = tracks

    # test written
    def get_max_click(self):
        res = 0
        for t in self.tracks:
            for iterable in t.all_iterables_with_time_events():
                for n in iterable:
                    res = max(res, n.click)
                    if hasattr(n, 'end'):
                        res = max(res, n.end)
        for m in self.markers:
            res = max(res, m.click)

        return res

    # noinspection DuplicatedCode
    def __repr__(self):
        res = ['MidiSong cpq:{}'.format(self.cpq)]
        res.extend(['{}'.format(t) for t in self.time_signatures])
        res.extend(['{}'.format(t) for t in self.tempo_changes])
        res.extend(['{}'.format(t) for t in self.tracks])

        return ' '.join(res)

    def enlarge_cpq_by_factor(self, factor):
        """alters cpq and all events in place"""
        self.cpq *= factor
        for L in self.all_iterables_with_time_events():
            for evt in L:
                # enlarge end before click, because sometimes enlarging click also automatically enlarges end.
                # (We wouldn't want to enlarge end twice.)
                if hasattr(evt, 'end'):
                    evt.end *= factor
                evt.click *= factor
        self.max_click_on_init *= factor

    def dump(self, filename=None, file=None):
        return _dump(cpq=self.cpq,
                     max_click=self.get_max_click(),
                     time_sigs=self.time_signatures,
                     tempo_changes=self.tempo_changes,
                     markers=self.markers,
                     tracks=self.tracks,
                     filename=filename,
                     file=file)

    # test written
    def quantize_notes(self, q=(4, 3), quantize_end_too=True):
        """in place operation.

        You probably *don't* actually want to use this function. It is much more likely that you want to convert
        this into a MidiSongByMeasure (using MidiSongByMeasure.from_MidiSong), and then run
        quantize_notes_by_measure on that MidiSongByMeasure."""
        lcm_q = extended_lcm(q)
        if self.cpq % lcm_q != 0:
            warning = 'WARNING: cpq={} is incompatible with quantize q={}. Stretching all events in object by factor={} prior to quantizing, including max_click_on_init. This warning will not be shown again this session.'.format(
                self.cpq, q, lcm_q)
            warning_type = 'WARNING: quantize_notes'
            if warning_type not in _WARNINGS_PRINTED:
                print(warning)
                _WARNINGS_PRINTED.add(warning_type)
            self.enlarge_cpq_by_factor(lcm_q)

        for t in self.tracks:
            t.quantize_notes(cpq=self.cpq, q=q, quantize_end_too=quantize_end_too)

    # test written
    def transpose(self, amt):
        """Transposes all non-drum tracks by amt. In place operation."""
        for t in self.tracks:
            if not t.is_drum:
                t.transpose(amt=amt)

    def to_deduping_str(self):
        """note onset chromagram, basically"""
        T = combine_tracks([t for t in self.tracks if not t.is_drum])  # pitched tracks only
        pitch_classes_at_click = collections.defaultdict(set)
        clicks = set()
        for n in T.notes:
            pitch_classes_at_click[n.click].add(n.pitch % 12)
            clicks.add(n.click)
        clicks = sorted(list(clicks))
        res = ''
        for c in clicks:
            pcs = sorted(list(pitch_classes_at_click[c]))
            res += ';c:{}p:{}'.format(c, pcs)
        return res

    def de_subsetting_set(self, instrument_agnostic=True):
        res = set()
        for T in self.tracks:
            pitch_classes_at_click = collections.defaultdict(set)
            clicks = set()
            for note in T.notes:
                pitch_classes_at_click[note.click].add(note.pitch)
                clicks.add(note.click)
            clicks = sorted(list(clicks))
            s = '' if instrument_agnostic else str(T.inst)
            for c in clicks:
                pitch_classes = sorted(list(pitch_classes_at_click[c]))
                s += ';c:{}p:{}'.format(c, pitch_classes)
            res.add(s)
        return res

    # TODO - write test / use this for some new functionality (e.g., arranging)
    def piano_reduction(self):
        """Creates a new MidiSong with a single pitched track that results from combining all pitched tracks in self.
        The resulting MidiSong also has drum tracks which are straight copies of the drum tracks in self.

        Returns a new MidiSong. Does not alter self."""
        res = copy.copy(self)
        to_combine = [t for t in self.tracks if not t.is_drum]
        new_t = combine_tracks(to_combine, inst_for_result=0)
        d_tracks = [t for t in res.tracks if t.is_drum]
        res.tracks = [new_t]
        res.tracks.extend(d_tracks)
        return res


# Note that unlike time_signatures for MidiSong's, a MidiSongByMeasure S may have TempoChange's with equal values
# next to each other in S.tempo_changes.
class MidiSongByMeasure(object):
    @classmethod
    def from_MidiSong(cls, S: MidiSong, measure_endpoints: list = None, consume_calling_song=False):
        """If measure_endpoints is None, then measure_endpoints will be computed from the time sigs in S.
        You can just leave measure_endpoints as None unless you want to use a set of custom measure endpoints.

        If consume_calling_song, then you should never reuse the song input to this function. Use
        consume_calling_song = False to ensure that the calling song is not altered by this function."""
        if not consume_calling_song:
            S = copy.copy(S)

        if measure_endpoints is None:
            measure_endpoints = measure_endpoints_from_time_sigs(cpq=S.cpq, max_click=S.get_max_click(),
                                                                 time_signatures=S.time_signatures)

        tracks = []
        for t in S.tracks:
            # S itself was already copied above if necessary, so we can use consume_calling_track=True here
            t_new = ByMeasureTrack.from_Track(track=t, measure_endpoints=measure_endpoints,
                                              consume_calling_track=True)
            tracks.append(t_new)

        return cls(tracks=tracks, measure_endpoints=measure_endpoints, markers=S.markers,
                   tempo_changes=S.tempo_changes, cpq=S.cpq)

    def __init__(self, tracks=None, measure_endpoints=None, markers=None, tempo_changes=None, cpq=960):
        """You are probably best off using .from_MidiSong rather than using this initializer directly.

        :type tracks: list[ByMeasureTrack]
        :type measure_endpoints: list[int]
        :type markers: list[Marker]
        :type tempo_changes: list[TempoChange]
        :type cpq: int
        """

        self.tracks = tracks if tracks is not None else []
        self.__measure_endpoints = copy.copy(measure_endpoints) if measure_endpoints is not None else []
        self.__measure_endpoints.sort()
        if len(self.__measure_endpoints) != len(set(self.__measure_endpoints)):
            raise ValueError('measure_endpoints contains repeated value: {}'.format(self.__measure_endpoints))
        if len(self.__measure_endpoints) <= 1:
            raise ValueError('measure endpoints cannot be a list of length <=1 (which represents 0 measures).')

        self.markers = markers if markers is not None else []

        self.tempo_changes = tempo_changes if tempo_changes is not None else []
        self.__fix_tempo_changes()

        self.cpq = cpq

    # noinspection DuplicatedCode
    def __fix_tempo_changes(self):
        self.tempo_changes.sort(key=lambda x: x.click)
        self.tempo_changes = remove_duplicate_events_at_same_click(self.tempo_changes, cmp=lambda x: x.click)
        if not self.tempo_changes:
            self.tempo_changes = [TempoChange(val=120, click=0)]
        if self.tempo_changes[0].click > 0:
            self.tempo_changes.insert(0, TempoChange(val=120, click=0))

    def __copy__(self):
        tracks = [copy.copy(x) for x in self.tracks]
        markers = [copy.copy(x) for x in self.markers]
        tempo_changes = [copy.copy(x) for x in self.tempo_changes]
        return type(self)(tracks=tracks, measure_endpoints=self.__measure_endpoints, markers=markers,
                          tempo_changes=tempo_changes, cpq=self.cpq)

    def sort_tracks_by_inst(self):
        """in place operation"""
        self.tracks.sort(key=lambda t: t.inst if t.inst is not None else -1000)  # "None" insts go first

    # test written
    def sort_tracks_by_inst_and_avg_note_pitch(self):
        """in place operation. Looks only at note_ons, not notes or note_offs."""
        def sort_key(tr: "ByMeasureTrack"):
            notes = []
            for t_m in tr.tracks_by_measure:
                notes.extend(t_m.note_ons)
            p = average_pitch(notes)
            return -1 * round(p, 2)

        self.tracks = _track_sorter_by_inst_and_avg_note_pitch(tracks=self.tracks, sort_key_fn=sort_key)

    def get_measure(self, measure_idx=0) -> "list[Track]":
        return [t[measure_idx] for t in self.tracks]

    def get_measures(self):
        res = [self.get_measure(i) for i in range(len(self.__measure_endpoints) - 1)]
        return res

    def get_measure_slice(self, st, end):
        end = min(end, self.get_n_measures())
        res = [self.get_measure(i) for i in range(st, end)]
        return res

    def get_n_measures(self):
        return len(self.__measure_endpoints) - 1

    def get_track(self, track_idx):
        return self.tracks[track_idx]

    def get_measure_lengths(self):
        res = []
        for i, val in enumerate(self.__measure_endpoints):
            if i:
                prev = self.__measure_endpoints[i - 1]
                res.append(val - prev)
        return res

    def get_measure_endpoints(self, make_copy=True):
        """DO NOT ALTER THE RESULT (unless make_copy=True)"""
        if make_copy:
            return copy.copy(self.__measure_endpoints)
        else:
            return self.__measure_endpoints

    # test written
    def get_tempo_at_start_of_each_measure(self) -> "list[float]":
        """Returns a list of length equal to self.get_n_measures(). self.tempo_changes must be sorted by click for
        this to work."""
        res = []
        tc_clicks = [x.click for x in self.tempo_changes]
        for me in self.__measure_endpoints:
            pos = bisect.bisect_right(tc_clicks, me) - 1
            res.append(self.tempo_changes[pos].val)
        res.pop(-1)
        return res

    @abstractmethod
    def all_iterables_with_time_events(self):
        """must return a list of lists"""
        res = []
        res.append(self.markers)
        res.append(self.tempo_changes)
        for T in self.tracks:
            for iterable in T.all_iterables_with_time_events():
                res.append(iterable)
        return res

    def remove_tracks_with_no_notes(self):
        """in place operation"""
        tracks = []
        for t in self.tracks:
            if t.has_notes():
                tracks.append(t)
        self.tracks = tracks

    def __repr__(self):
        res = ['MidiSongByMeasure cpq:{} MEs:'.format(self.cpq)]
        res.extend(['{}'.format(t) for t in self.__measure_endpoints])
        res.extend(['{}'.format(t) for t in self.tempo_changes])
        res.extend(['{}'.format(t) for t in self.tracks])
        return ' '.join(res)

    def enlarge_cpq_by_factor(self, factor):
        """alters cpq, measure_endpoints, and all events in place"""
        self.cpq *= factor
        for L in self.all_iterables_with_time_events():
            for evt in L:
                # enlarge end before click, because sometimes enlarging click also automatically enlarges end.
                # (We wouldn't want to enlarge end twice.)
                if hasattr(evt, 'end'):
                    evt.end *= factor
                evt.click *= factor
        self.__measure_endpoints = [x * factor for x in self.__measure_endpoints]

    def dump(self, filename=None, file=None):
        out_song = MidiSong.from_MidiSongByMeasure(self, consume_calling_song=False)
        out_song.dump(filename=filename, file=file)

    def quantize_notes_by_measure(self, q=(4, 3)):
        """in place operation. Returns True if this added a measure to self, and false otherwise."""

        lcm_q = extended_lcm(q)
        if self.cpq % lcm_q != 0:
            warning = 'WARNING: cpq={} is incompatible with quantize q={}. Stretching all events in object by factor={} prior to quantizing. This warning will not be shown again this session.'.format(
                self.cpq, q, lcm_q)
            warning_type = 'WARNING: quantize_notes_by_measure'
            if warning_type not in _WARNINGS_PRINTED:
                print(warning)
                _WARNINGS_PRINTED.add(warning_type)
            self.enlarge_cpq_by_factor(lcm_q)

        need_to_add_measure_endpoint = False
        for t in self.tracks:
            f_res = t.quantize_notes_by_measure(cpq=self.cpq, q=q, measure_endpoints=self.__measure_endpoints)
            need_to_add_measure_endpoint = need_to_add_measure_endpoint or f_res
            # this might change self__measure_endpoints.

        if need_to_add_measure_endpoint:
            # then add a measure endpoint
            self.extend_one_measure_to_the_right()

        return need_to_add_measure_endpoint

    def extend_one_measure_to_the_right(self):
        self.__measure_endpoints.append(2 * self.__measure_endpoints[-1] - self.__measure_endpoints[-2])

        # and add a measure to all elements of self.tracks that need one
        for t in self.tracks:
            if len(t.tracks_by_measure) != len(self.__measure_endpoints) - 1:
                t.tracks_by_measure.append(Track(inst=t.inst))

    def transpose(self, amt):
        """Transposes all non-drum tracks by amt. In place operation."""
        for t in self.tracks:
            if not t.is_drum:
                t.transpose(amt=amt)

    def change_cpq(self, to_cpq):
        """in place operation"""
        factor = to_cpq / self.cpq
        # self.quantize_notes_by_measure(q=(to_cpq, ))  # handles adding a measure endpoint at the end if necessary

        for t in self.tracks:
            t.change_cpq(from_cpq=self.cpq, to_cpq=to_cpq)

        # then update self.markers, self.cpq, self.measure_endpoints, and self.tempo_changes
        for m in self.markers:
            m.click = round(m.click * factor)

        for m in self.tempo_changes:
            m.click = round(m.click * factor)

        self.cpq = to_cpq

        lengths = self.get_measure_lengths()
        lengths = [max(1, round(x * factor)) for x in lengths]  # min length = 1
        self.__measure_endpoints = [self.__measure_endpoints[0]]
        for x in lengths:
            self.__measure_endpoints.append(self.__measure_endpoints[-1] + x)

    def get_measure_indexes_containing_no_note_ons(self):
        def is_empty_measure(M):
            for M_t in M:
                if M_t.note_ons:
                    return False
            return True

        empty = []
        for i, m in enumerate(self.get_measures()):
            if is_empty_measure(m):
                empty.append(i)
        return empty

    # test written
    def get_empty_measure_indexes(self):
        def is_empty_measure(M):
            for M_t in M:
                if not M_t.is_empty():
                    return False
            return True

        empty = []
        for i, m in enumerate(self.get_measures()):
            if is_empty_measure(m):
                empty.append(i)
        return empty

    # test written
    def get_measure_index_and_click_in_measure(self, song_click):
        measure_i = bisect.bisect_right(self.__measure_endpoints, song_click) - 1
        click_in_measure = song_click - self.__measure_endpoints[measure_i]
        return [measure_i, click_in_measure]

    # test written
    def get_song_click(self, measure_index, click_in_measure):
        return self.__measure_endpoints[measure_index] + click_in_measure

    # helper function for remove_measures_by_index
    def __step_1_remove_measures_by_index_marker_treatment(self, handle, L):
        """'handle' a list of events, like self.markers. Alters handle in place, among other
        things. L the list of measure indexes to be removed."""
        marker_info = []
        for m in handle:
            # append to marker_info (measure number, click in that measure)
            marker_info.append(self.get_measure_index_and_click_in_measure(m.click))
        # then adjust marker measures and remove markers
        n_popped = 0
        original_marker_info = [copy.copy(x) for x in marker_info]
        popped = []
        for i in L:
            for j, mi in enumerate(marker_info):
                if original_marker_info[j][0] > i:
                    mi[0] -= 1  # decrement measure numbers from markers that are going to get moved left
                elif original_marker_info[j][0] == i:
                    mi[0] = -1  # set to -1 measure numbers for markers that are getting removed
                    popped.append(handle.pop(j - n_popped))
                    n_popped += 1
        return marker_info

    # helper function for remove_measures_by_index
    def __step_1_remove_measures_by_index_tempo_change_treatment(self, handle, L):
        """'handle' a list of events, like self.tempo_changes. Alters handle in place, among other
        things. L the list of measure indexes to be removed."""
        handle_info = []
        for m in handle:
            handle_info.append(self.get_measure_index_and_click_in_measure(m.click))

        original_handle_info = [copy.copy(x) for x in handle_info]
        for i in L:
            for j, mi in enumerate(handle_info):
                if original_handle_info[j][0] == i:
                    mi[1] = 0
                elif original_handle_info[j][0] > i:
                    mi[0] -= 1  # decrement measure numbers from tempo changes that are getting moved left
        return handle_info

    # helper function for remove_measures_by_index
    def __step_2_remove_measures_by_index(self, handle, handle_info):
        """'handle' a list of events, like L=self.markers. Alters handle in place."""
        marker_i = -1
        for info in handle_info:
            if info[0] > -1:
                marker_i += 1
                handle[marker_i].click = self.__measure_endpoints[info[0]] + info[1]

    # test written
    def remove_measures_by_index(self, L):
        """L a list of integers (measure indexes). In-place operation."""

        # handle some annoying stuff for markers
        marker_info = self.__step_1_remove_measures_by_index_marker_treatment(handle=self.markers, L=L)
        tc_info = self.__step_1_remove_measures_by_index_tempo_change_treatment(handle=self.tempo_changes, L=L)

        # the big part of this function
        for t in self.tracks:
            t.remove_measures_by_index(L)

        # update measure endpoints using measure lengths
        lengths = self.get_measure_lengths()
        for i, ct in enumerate(L):
            lengths.pop(ct - i)
        self.__measure_endpoints = [0]
        for x in lengths:
            self.__measure_endpoints.append(self.__measure_endpoints[-1] + x)

        # finally, set marker clicks and tempo change clicks using new measure endpoints
        self.__step_2_remove_measures_by_index(handle=self.markers, handle_info=marker_info)
        for tc, tc_info in zip(self.tempo_changes, tc_info):
            tc.click = self.get_song_click(measure_index=tc_info[0], click_in_measure=tc_info[1])
        # if this moves tempo changes on top of each other, get rid of all but the last one.
        self.__fix_tempo_changes()

    # test written
    def remove_empty_measures_at_beginning_and_end(self):
        """in place operation"""
        empty_indexes = self.get_empty_measure_indexes()
        to_remove = set()
        for i, idx in enumerate(empty_indexes):
            if i != idx:
                break
            else:
                to_remove.add(idx)

        for i, idx in enumerate(reversed(empty_indexes)):
            if idx != self.get_n_measures() - 1 - i:
                break
            else:
                to_remove.add(idx)

        to_remove = sorted(list(to_remove))
        self.remove_measures_by_index(L=to_remove)

    # test written
    def remove_every_empty_measure_that_has_an_empty_preceding_measure(self):
        EMIs = self.get_empty_measure_indexes()
        to_del = []
        for i, emi in enumerate(EMIs):
            if i:
                prev = EMIs[i - 1]
                if emi - 1 == prev:
                    to_del.append(emi)
        if to_del:
            self.remove_measures_by_index(L=to_del)

    # test written
    def find_one_pair_of_non_overlapping_track_consolidation_indexes(self) -> "Tuple[int, int] or None":
        insts = collections.Counter()
        for t in self.tracks:
            insts[t.inst] += 1
        # insts is the set of instruments that appear in > 1 track
        insts = set(inst for inst in insts if insts[inst] > 1)

        def no_overlap(L1, L2):
            """L1, L2 0-1 lists of the same length"""
            for x, y in zip(L1, L2):
                if x * y:
                    return False
            return True

        for inst in insts:
            tracks = [(i, t) for i, t in enumerate(self.tracks) if t.inst == inst]
            coverages = [(i, t.get_note_on_note_off_measure_coverage()) for i, t in tracks]
            for i, T1 in enumerate(coverages):
                for T2 in coverages[i+1:]:
                    if no_overlap(T1[1], T2[1]):
                        return T1[0], T2[0]
        return None

    # test written
    def is_octave_collapse_of_some_track_in_this_measure(self, tr_i: int, measure_i: int):
        """Returns False if tr_i is a drum track.
        Otherwise, returns True iff tr_i is an octave shift of some other measure in measure_i"""
        if self.tracks[tr_i].is_drum:
            return False

        notes_by_click_and_octave = set()
        tr = self.tracks[tr_i]
        for n in tr.tracks_by_measure[measure_i].note_ons:
            notes_by_click_and_octave.add((n.click, n.pitch % 12))

        for i, tr in enumerate(self.tracks):
            if i != tr_i and not tr.is_drum:
                this_tr_notes_by_click_and_octave = set()
                for n in tr.tracks_by_measure[measure_i].note_ons:
                    this_tr_notes_by_click_and_octave.add((n.click, n.pitch % 12))
                if this_tr_notes_by_click_and_octave == notes_by_click_and_octave:
                    return True

        return False

    # test written
    def is_poly(self, track_idx, measure_idx):
        """returns True iff the track track_idx at measure measure_idx has a click c such that there are two note
        onsets at c."""
        tr = self.tracks[track_idx]
        tr = tr.tracks_by_measure[measure_idx]
        return tr.is_poly()

    # test written
    def horiz_note_onset_density(self, tr_i: int,
                                 measures: typing.Iterable[int],
                                 denominator_includes_only_measures_with_note_ons=True) -> float or None:
        return _horiz_note_onset_density(S=self, tr_i=tr_i, measures=measures,
               denominator_includes_only_measures_with_note_ons=denominator_includes_only_measures_with_note_ons)

    # test written
    def vert_note_onset_density(self, tr_i: int,
                                measures: typing.Iterable[int]) -> float or None:
        """assumes tr_i is sorted"""
        return _vert_note_onset_density(S=self, tr_i=tr_i, measures=measures)

    # test written
    def pitch_interval_hist(self, tr_i: int,
                            measures: typing.Iterable[int]) -> dict[float, int]:
        """assumes tr_i is sorted"""
        return _pitch_interval_hist(S=self, tr_i=tr_i, measures=measures)

    # test written
    def consolidated_pitch_interval_hist(self, tr_i: int,
                                         measures: typing.Iterable[int]) -> dict[str, float]:
        """assumes tr_i is sorted"""
        return _consolidated_pitch_interval_hist(S=self, tr_i=tr_i, measures=measures)

    # test written
    def vert_note_onset_n_pitch_classes_avg(self, tr_i: int,
                                            measures: typing.Iterable[int]) -> float or None:
        """assumes tr_i is sorted"""
        return _vert_note_onset_n_pitch_classes_avg(S=self, tr_i=tr_i, measures=measures)

    # test written
    def horiz_note_onset_irregularity(self, tr_i: int,
                                      measures: typing.Iterable[int]) -> float:
        return _horiz_note_onset_irregularity(S=self, tr_i=tr_i, measures=measures)

    def horiz_note_onset_irregularity_new_idea(self, tr_i: int,
                                               measures: typing.Iterable[int],
                                               use_np=False) -> float:
        return _horiz_note_onset_irregularity_new_idea(S=self, tr_i=tr_i, measures=measures, use_np=use_np)

    # def horiz_note_onset_by_measure_stdev(self,
    #                                       tr_i: int,
    #                                       measures: typing.Iterable[int]) -> float or None:
    #     return _horiz_note_onset_by_measure_stdev(S=self, tr_i=tr_i, measures=measures)

    def pitch_range(self, tr_i: int, measures: typing.Iterable[int]) -> tuple[int, int] or None:
        """returns (lowest note_on pitch, highest note_on pitch) in the specified track-measures.
        Returns None of there are no note_on's in the specified track-measures."""
        tr = self.tracks[tr_i]
        pitches = set()
        for m_i in measures:
            for n in tr.tracks_by_measure[m_i].note_ons:
                pitches.add(n.pitch)
        try:
            hi = max(pitches)
            lo = min(pitches)
        except ValueError:
            return None
        return lo, hi


# for both MidiSong and MidiSongByMeasure
def _track_sorter_by_inst_and_avg_note_pitch(tracks, sort_key_fn):
    by_inst = collections.defaultdict(list)
    for t in tracks:
        by_inst[t.inst].append(t)
    for inst, L in by_inst.items():
        L.sort(key=sort_key_fn)

    insts = sorted(list(by_inst.keys()), key=lambda k: k if k is not None else -1000)  # "None" insts go first
    new_tracks = []
    for inst in insts:
        for t in by_inst[inst]:
            new_tracks.append(t)
    return new_tracks


##################################################
# Functions
def _index_of_closest_element_in_sorted_numeric_list(L: list, x) -> int:
    """returns i such that abs(x - L[i]) is minimized. If there is a tie, defaults to the rightmost index."""
    i = bisect.bisect_right(L, x)
    if i == 0:
        return i
    elif i == len(L):
        return i - 1
    elif L[i - 1] == x:
        return i - 1
    else:
        a, b = L[i - 1], L[i]
        if x - a < b - x:
            return i - 1
        else:
            return i


def _lcm(a: int, b: int):
    return abs(a * b) // math.gcd(a, b)


def extended_lcm(L: list or tuple):
    """L a list of integers"""
    res = 1
    for x in L:
        res = _lcm(res, x)
    return res


def _dump(cpq: int, max_click: int, time_sigs, markers, tracks, tempo_changes, filename=None, file=None):
    """
    :type time_sigs: list[TimeSig]
    :type markers: list[Marker]
    :type tracks: list[Track]
    :type tempo_changes: list[TempoChange]
    :type filename: str
    :type file: str
    """
    import miditoolkit
    # use miditoolkit
    res = miditoolkit.MidiFile(ticks_per_beat=cpq)

    # translate our info back to miditoolkit values
    res.max_tick = max_click
    res.time_signature_changes = [miditoolkit.TimeSignature(numerator=x.num, denominator=x.denom, time=x.click) for x in
                                  time_sigs]
    res.tempo_changes = []
    res.markers = [miditoolkit.Marker(text=x.text, time=x.click) for x in markers]
    res.tempo_changes = [miditoolkit.TempoChange(tempo=x.val, time=x.click) for x in tempo_changes]

    res.instruments = []
    for t in tracks:

        # first, fix up notes whose start and end clicks are the same, bc miditoolkit doesn't like that.
        t = copy.copy(t)
        for n in t.notes:
            if n.click == n.end:
                n.end += 1
        # this can revert a note to having the same start and end click, but it's the best we can do.
        fix_note_overlaps(t.notes)

        inst = t.inst
        if inst > 127:
            inst = 0
        elif inst is None:
            # TODO split into a track for each instrument, etc.
            raise NotImplementedError('cannot currently write track with None instrument to midi')
        their_track = miditoolkit.Instrument(program=inst, is_drum=t.is_drum)
        their_track.control_changes = [miditoolkit.ControlChange(number=x.cc, value=x.val, time=x.click) for x in t.ccs]
        their_track.pitch_bends = [miditoolkit.PitchBend(pitch=x.val, time=x.click) for x in t.pitch_bends]

        # TODO: account for note_ons and note_offs too. (You never know: A track could have nonempty .notes
        #  as well as nonempty .note_ons and .note_offs.)
        their_track.notes = [miditoolkit.Note(velocity=x.vel, pitch=x.pitch, start=x.click, end=x.end) for x in t.notes]
        their_track.pedals = [miditoolkit.Pedal(start=x.click, end=x.end) for x in t.pedals]
        their_track.name = '{}'.format(t.inst)
        res.instruments.append(their_track)

    return res.dump(filename=filename, file=file)


# test written
def remove_duplicate_events_at_same_click(L: list, cmp=lambda x: x) -> list:
    """L a list of basic events (e.g., ccs or notes), sorted by click.
    If two or more events in L have the same .click property and compare equal (according to the supplied cmp function),
    then remove all but the last one of them in L.
    Outputs of the cmp function must be hashable.
    Returns a new list. (Does not alter L.)"""
    res = []
    by_click = collections.defaultdict(list)
    for evt in L:
        by_click[evt.click].append(evt)

    res_by_click = collections.defaultdict(list)

    for k in sorted(by_click.keys()):
        seen_at_this_click = set()
        for evt in reversed(by_click[k]):
            if cmp(evt) not in seen_at_this_click:
                seen_at_this_click.add(cmp(evt))
                res_by_click[k].append(evt)

    for k in sorted(res_by_click.keys()):
        for evt in reversed(res_by_click[k]):
            res.append(evt)

    return res


def _get_quantize_endpoints(cpq: int, q=(4, 3)):
    quantize_endpoints = []
    for t in q:
        for a in range(t + 1):
            quantize_endpoints.append(int(a * cpq / t))
    quantize_endpoints = sorted(list(set(quantize_endpoints)))  # per quarter note
    return quantize_endpoints


def _get_all_grid_points_per_qn(cpq: int, q=(4, 3)):
    all_grid_points_per_qn = []
    denom = extended_lcm(q)
    for a in range(denom):
        all_grid_points_per_qn.append(int(a * cpq / denom))
    return all_grid_points_per_qn


# test written
def quantize_list(L, start_click, cpq, end_click=None, q=(4, 3)):
    """L a list of events whose .click values are start_click or later.

    Alters the .click values of the elements of L in place according to the supplied cpq and q values.
    Note that the elements of L are altered in place, and that only their .click (not .end) values are altered directly.

    The most common use for this function is when the supplied list L represents a measure, start_click is the
    start click of that measure, and end_click is the end click of that measure.

    Note that this function may cause an event to quantize to end_click or to a value larger than end_click. The return
    value of this function is the list of indexes in L of such elements, after having been quantized. (This is the
    only way that end_click is used in the function.)
    Such events are still in L, but you can use the the return value of this function to pop them from L if you want to.

    :type L: list
    :type start_click: int
    :type cpq: int
    :type end_click: int or None
    :type q: tuple
    """

    lcm_q = extended_lcm(q)
    if cpq % lcm_q != 0:
        raise ValueError(
            'cpq={} is not compatible with q={} because cpq={} is not divisible by {}.'.format(cpq, q, cpq, lcm_q))

    quantize_endpoints = _get_quantize_endpoints(cpq=cpq, q=q)
    # print(quantize_endpoints)

    res = []

    for i, evt in enumerate(L):
        click = evt.click - start_click
        if click < 0:
            raise ValueError('click={} in L before start_click={}'.format(evt.click, start_click))
        y_i = _index_of_closest_element_in_sorted_numeric_list(quantize_endpoints, click % cpq)
        y = quantize_endpoints[y_i]

        click_quantized = cpq * int(click / cpq) + y  # click's QN start (which is why we round down) + offset
        evt.click = click_quantized + start_click
        if end_click is not None and evt.click >= end_click:
            res.append(i)

    return res


# test written
def apply_pedals_to_extend_note_lengths(L, pedals):
    """L a list of notes, sorted by click; pedals a list of Pedal events, sorted by click.
    Alters L and the elements of L in place by altering their .end values.
    It is highly recommended to call fix_note_overlaps(L) after running this function.

    :type L: list[Notes]
    :type pedals: list[Pedals]
    """
    for note in L:
        for p in pedals:
            if p.click <= note.end <= p.end:
                note.end = p.end


# test written
def fix_note_overlaps(L):
    """L a list of notes, sorted by click.
    Alters L and the elements of L in place to guarantee that if N1 precedes N2 in L and N1.pitch == N2.pitch, then
    N1.end <= N2.click

    :type L: list[Note]"""
    by_pitch = collections.defaultdict(list)
    for note in L:
        by_pitch[note.pitch].append(note)

    for pitch in by_pitch:
        notes_at_this_pitch = by_pitch[pitch]
        for i, note in enumerate(notes_at_this_pitch):
            if i > 0:
                prev = notes_at_this_pitch[i - 1]
                if prev.end > note.click:
                    prev.end = note.click


# test written
def combine_tracks(L, inst_for_result=None) -> Track:
    """L a list of Tracks. If inst_for_result is None, uses the inst of the first track in L (which itself may be None).

    Adds all of the elements in each list in L to the result, then fixes any event collisions/overlaps that result.
    Notes with the same pitch and click values are considered to collide, even if they have different inst's.
    Similar rules apply to ccs, pitch bends, and pedals (in that .inst values are not considered when determining
    collisions).

    When notes collide, the longest note is kept.

    Returns a new, sorted Track.

    Elements (such as notes) of this track are the same in-memory elements of those in the tracks in L.

    The result of this function will have empty .note_ons and .note_offs, but will use .note_ons and .note_offs from
    tracks in L. Such lists of NoteOn's and NoteOff's must be convertible back into Note's via the function
    compute_join_note_ons_and_note_offs_to_notes for this function to work.

    :type L:list[Track]
    :type inst_for_result: int"""
    if inst_for_result is None and L:
        inst_for_result = L[0].inst

    notes = []
    ccs = []
    pbs = []
    pedals = []
    extra_info = {}
    for track in L:
        notes.extend(track.notes)
        if track.note_ons:
            more_notes = compute_join_note_ons_and_note_offs_to_notes(track.note_ons, track.note_offs)
            notes.extend(more_notes)
        ccs.extend(track.ccs)
        pbs.extend(track.pitch_bends)
        pedals.extend(track.pedals)
        extra_info.update(track.extra_info)

    notes.sort(key=lambda x: x.end - x.click)  # make it so that the longest note at a click is kept.
    notes.sort(key=lambda x: x.click)
    ccs.sort(key=lambda x: x.click)
    pbs.sort(key=lambda x: x.click)
    pedals.sort(key=lambda x: x.click)

    notes = remove_duplicate_events_at_same_click(notes, cmp=lambda x: x.pitch)
    ccs = remove_duplicate_events_at_same_click(ccs, cmp=lambda x: x.cc)
    pbs = remove_duplicate_events_at_same_click(pbs, cmp=lambda x: x.click)
    pedals = remove_duplicate_events_at_same_click(pedals, cmp=lambda x: x.end)

    # and fix overlapping notes
    fix_note_overlaps(notes)

    res = Track(inst=inst_for_result, notes=notes, ccs=ccs, pitch_bends=pbs, pedals=pedals, extra_info=extra_info)
    res.sort()
    return res


# test written
def measure_endpoints_from_time_sigs(cpq, max_click, time_signatures, max_measures=100000) -> list:
    """time signatures a list of TimeSig's sorted by click, whose first element has click 0

    :type cpq: int
    :type max_click: int
    :type time_signatures: list[TimeSig]
    :type max_measures: int
    """
    if not time_signatures:
        raise ValueError('time_signatures must be nonempty')

    if time_signatures[0].click != 0:
        raise ValueError('first time signature has click > 0: {}'.format(time_signatures[0]))

    time_signatures = [x for x in time_signatures if x.click <= max_click]

    res = []
    for i, time_sig in enumerate(time_signatures):
        n_qn_per_measure = 4 * time_sig.num / time_sig.denom
        if i < len(time_signatures) - 1:
            next_time_sig = time_signatures[i + 1]
            next_sig_click = next_time_sig.click
        else:
            next_sig_click = max_click + 1
        measure_click = time_sig.click
        res.append(measure_click)
        if len(res) - 1 > max_measures:
            raise ValueError('Too many measures')

        next_measure_click = measure_click + int(cpq * n_qn_per_measure)
        while next_measure_click < next_sig_click:
            measure_click = next_measure_click
            res.append(measure_click)
            next_measure_click = measure_click + int(cpq * n_qn_per_measure)
            if len(res) - 1 > max_measures:
                raise ValueError('Too many measures')

    # Add one more measure (i.e., measure click) at the end.
    # There's always at least one time sig so you can ignore the PyCharm warning.
    n_qn_per_measure = 4 * time_sig.num / time_sig.denom
    next_measure_click = measure_click + int(cpq * n_qn_per_measure)
    res.append(next_measure_click)
    return res


# time sig functions
# test written
def _delete_equivalent_consecutive_time_sigs(time_signatures):
    """time_signatures a list of TimeSig's. Modifies this list in place."""
    to_pop = []
    for i, TS in enumerate(time_signatures):
        if i:
            prev = time_signatures[i - 1]
            if TS.is_equiv_to(prev):
                to_pop.append(i)
    for i, pop_i in enumerate(to_pop):
        time_signatures.pop(pop_i - i)


# test written
def _lengthen_time_sigs_where_possible(time_signatures, cpq, from_=(1, 4), to_=(2, 4)):
    """time_signatures a list of TimeSig's, sorted by click. Modifies this list in place."""
    to_insert = []
    for i, e in enumerate(time_signatures):
        if i < len(time_signatures) - 1:
            next_ts = time_signatures[i + 1]
        else:
            next_ts = None

        if (e.num, e.denom) == from_:
            if next_ts is None:
                # then go ahead and change this one
                e.num, e.denom = to_
            else:
                # then we need to make sure we are able to change this one
                n_clicks = next_ts.click - e.click
                clicks_per_target_measure = 4 * to_[0] / to_[1] * cpq
                r, q = math.modf(n_clicks / clicks_per_target_measure)
                n_measures_target = int(q)
                # if we can fit at least one measure of the to_ time sig
                if n_measures_target > 0:
                    # then change e accordingly
                    e.num, e.denom = to_
                    # this can leave a fractional measure before the next time sig, though, so we should add a new time sig
                    # of the from_ variety at the appropriate place
                if r > 0.00001:
                    to_insert.append([i + 1, TimeSig(num=from_[0], denom=from_[1],
                                                     click=e.click + round(q * clicks_per_target_measure))])
    for i in range(len(to_insert)):
        L = to_insert[i]
        insert_idx, e = L
        time_signatures.insert(insert_idx + i, e)


# helper for _standardize_time_sig_lengths
# test written
def _shorten_time_sigs_where_needed(time_signatures, cpq, final_click, too_long_qn=8):
    """time_signatures a list of TimeSig's, sorted by click. Modifies this list in place.
    For each time sig that is too_long_qn quarter notes or longer, we change time_signatures as necessary to
    replace every measure induced by that time sig with 4/4 + a bit more if needed.
    For instance, if one of the time signatures is 9/4, we would replace it with 4/4 for two measures, then 1/4
    for one quarter note, then 4/4 for another two measures, then 1/4 for another quarter note, etc.
    too_long_qn must be 4 or more."""
    if too_long_qn < 4:
        raise ValueError("too_long_qn must be 4 or larger. Got: {}".format(too_long_qn))

    to_insert = []
    for i, e in enumerate(time_signatures):
        if i < len(time_signatures) - 1:
            next_ts = time_signatures[i + 1]
        else:
            next_ts = None

        if e.num < 1 or e.denom < 1:
            raise ValueError('time signature invalid: {}'.format(e))

        if 4 * e.num / e.denom >= too_long_qn:
            # then, since too_long_qn >=4, we see that e.num/e.denom will be >=1. Hence q >= 1.
            # q is the number of measures of 4/4 we will replace this with.
            # r is the numerator for the measure of more stuff
            # running ex: if e is 17/8, then q = 2 and r = 1
            q, r = divmod(e.num, e.denom)

            if r == 0:
                # then just change e to 4/4, and we are done.
                e.num, e.denom = 4, 4

            else:
                # then we alternate between writing e.g. 4/4 (mode 0) and 1/8 (mode 1).
                # We change e itself to 4/4 after the loop.
                cmp = next_ts.click if next_ts is not None else final_click
                next_click = e.click + 4 * cpq * q  # advance
                mode = 1
                done = next_click >= cmp
                while not done:
                    if mode == 0:
                        to_insert.append([i + 1, TimeSig(num=4, denom=4, click=next_click)])
                        next_click += 4 * cpq * q
                        mode = 1

                    elif mode == 1:
                        to_insert.append([i + 1, TimeSig(num=r, denom=e.denom, click=next_click)])
                        next_click += int(4 * cpq * r / e.denom)
                        mode = 0

                    done = next_click >= cmp

                # replace this time sig with 4/4
                e.num, e.denom = 4, 4

    for i in range(len(to_insert)):
        L = to_insert[i]
        insert_idx, e = L
        time_signatures.insert(insert_idx + i, e)


# useful in its own right; helper for standardize_time_sigs
# test written
def _standardize_time_sig_lengths(time_signatures, cpq, final_click):
    """time_signatures a list of TimeSig's. Modifies the elements of L in place."""

    # first modify numerators that are multiples of 7, 3, 4, and 5, if they would cause measures to be >= 8 qn's.
    # For instance, this turns 14/4 into 7/4 and 15/4 into 5/4.
    for e in time_signatures:
        if e.num >= e.denom * 2:
            # standardize these to x/4 if possible
            while e.denom > 4 and e.num % 2 == 0:
                e.num //= 2
                e.denom //= 2
            # ensure that we are dealing with x/y, where y >= 4
            while e.denom < 4:
                e.num *= 2
                e.denom *= 2
            for k in [7, 3, 4, 5]:
                if e.num % k == 0:
                    e.num = k

    # next, standardize time sigs of the form pow_of_2 / pow_of_2, but not 2/4 or 4/4
    # for e in time_signatures:
    #     if abs(math.log(e.num, 2) - int(math.log(e.num, 2))) < .00001:
    #         e.num = e.num // min(e.num, e.denom)
    #         e.denom = e.denom // min(e.num, e.denom)

    # next, lengthen short time signatures wherever possible
    # largest_denom = 1
    # for e in time_signatures:
    #     largest_denom = max(largest_denom, e.denom)
    # denom = largest_denom
    # while denom > 1:
    #     # ex: lengthen from 1/64 to 1/32
    #     # for num_exp in range(math.log(denom, 2))
    #     # There's an issue here: This doesn't lengthen e.g., 2/32 like it would 1/16.
    #     _lengthen_time_sigs_where_possible(time_signatures=time_signatures, cpq=cpq,
    #                                        from_=(1, denom), to_=(1, denom // 2))
    #     _delete_equivalent_consecutive_time_sigs(time_signatures=time_signatures)
    #     denom = denom // 2

    # then shorten where necessary
    _shorten_time_sigs_where_needed(time_signatures=time_signatures, cpq=cpq, final_click=final_click, too_long_qn=8)
    _delete_equivalent_consecutive_time_sigs(time_signatures=time_signatures)

    # finally, put x/1 and y/2 back to z/4
    # for e in time_signatures:
    #     while e.denom < 4:
    #         e.num *= 2
    #         e.denom *= 2


# test written
def clean_up_time_sigs(time_signatures, cpq, final_click) -> list:
    """time_signatures a list of TimeSig's, sorted by click, with a TimeSig at click 0.

    final_click is only needed when the last TimeSig in time_signatures is "weird" and longer than 8 qn's, .e.g., 11/4.
    The reason is that this function will replace 11/4 with 2 measures of 4/4, then 1 measure of 3/4, then 2 measures of
    4/4, then 1 measure of 3/4, etc., and it needs to know when to stop.

    This function returns a new list containing some elements from time_signatures, which may or may not be modified,
    as well as some new TimeSig's. Hence you should not re-use the input time_signatures, or any of its elements,
    if you expect them to be the same after calling this function, because they probably won't be.

    Recommended usage: time_signatures = clean_up_time_sigs(time_signatures...).

    The output may still contain time signatures like 8/8 or 16/32.
    It may also induce different measures than the input.
    What is guaranteed is that every time sig will span less than 8 qn's.
    Currently, this function only shortens time sigs that span >= 8 qn's. Time sigs that span less than 8 qn's
    are left unchanged.

    To change time sigs that span less than 8 qns, edit the function _standardize_time_sig_lengths
    """

    # If we have more than one time sig at any given click, throw away all but the last one
    time_signatures = remove_duplicate_events_at_same_click(time_signatures, cmp=lambda x: x.click)

    # then standardize lengths
    _standardize_time_sig_lengths(time_signatures=time_signatures, cpq=cpq, final_click=final_click)

    # then if for some reason we have more than one time sig at any given click (due to rounding in the
    # above function), again throw away all but the last one
    time_signatures = remove_duplicate_events_at_same_click(time_signatures, cmp=lambda x: x.click)

    return time_signatures


# test written
def compute_split_notes_to_note_ons_and_note_offs(notes) -> list:
    """notes a list of Note objects. Returns the corresponding [note_ons, note_offs]. NoteOn and NoteOff objects
    will have their noteidx properties set for easy Note reconstruction"""
    note_ons, note_offs = [], []
    for noteidx_counter, note in enumerate(notes):
        inst = note.inst if hasattr(note, 'inst') else None
        on = NoteOn(pitch=note.pitch, vel=note.vel, click=note.click, inst=inst, noteidx=noteidx_counter)
        note_ons.append(on)
        off = NoteOff(pitch=note.pitch, click=note.end, inst=inst, noteidx=noteidx_counter)
        note_offs.append(off)
    return [note_ons, note_offs]


# test written
def compute_join_note_ons_and_note_offs_to_notes(note_ons, note_offs) -> list:
    notes = []
    events_by_idx = collections.defaultdict(list)
    for n in itertools.chain(note_ons, note_offs):
        events_by_idx[n.noteidx].append(n)
    noteidx_list = [noteidx for noteidx in events_by_idx]
    noteidx_list.sort()
    for noteidx in noteidx_list:
        NOn, NOff = events_by_idx[noteidx]
        inst = NOn.inst if hasattr(NOn, 'inst') else None
        notes.append(Note(pitch=NOn.pitch, vel=NOn.vel, click=NOn.click, end=NOff.click, inst=inst))
    return notes


# helper function for compute_time_signatures_from_measure_endpoints
def _time_sig_from_measure_length(x, cpq):
    iters = 0

    while x / cpq != round(x / cpq) and 4 * 2 ** iters < 64:
        iters += 1
        x *= 2

    if x / cpq != round(x / cpq):
        numerator = math.floor(x / cpq) + 1
    else:
        numerator = round(x / cpq)

    denominator = 4 * 2 ** iters
    while numerator % 2 == 0 and denominator % 2 == 0 and denominator > 4:
        numerator = numerator // 2
        denominator = denominator // 2

    return numerator, denominator


# test written
def compute_time_signatures_from_measure_endpoints(measure_endpoints, cpq):
    res = []
    for i, endpt in enumerate(measure_endpoints):
        if i:
            prev = measure_endpoints[i - 1]

            x = endpt - prev
            num, denom = _time_sig_from_measure_length(x, cpq)
            res.append(TimeSig(num=num, denom=denom, click=prev))
    _delete_equivalent_consecutive_time_sigs(res)
    return res


# test written
def transpose_list(L, amt):
    """L a list of objects with .pitch values. Alters all their pitch values by amt. Happens in place."""
    if amt:
        for evt in L:
            evt.pitch += amt


def compute_onset_polyphony(L):
    """L a list of Note's or NoteOn's. Returns None if L is empty; otherwise, returns a float in [0.0, 1.0].
    The output is simply the percentage of .click values in L that have >1 element in L with that .click value."""
    clicks = set()
    polyphonic_clicks = set()
    for note in L:
        if note.click in clicks:
            polyphonic_clicks.add(note.click)
        clicks.add(note.click)
    if L:
        return len(polyphonic_clicks)/len(clicks)
    else:
        return None


# test written
def average_pitch(L) -> float:
    """L a list of objects which have .pitch values. Returns 0.0 if L is empty."""
    if not L:
        return 0.0
    num = 0
    for n in L:
        num += n.pitch
    return num/len(L)


# new functions in v2:

# test written
def _set_to_list_of_contiguous_sub_lists(s: typing.Iterable[int]) -> typing.List[typing.List[int]]:
    s = list(s)
    s.sort()
    return _sorted_list_to_list_of_contiguous_sub_lists(s)


# test written (via testing _set_to_list_of_contiguous_sub_lists)
def _sorted_list_to_list_of_contiguous_sub_lists(lst: typing.List[int]) -> typing.List[typing.List[int]]:
    res = []
    this_lst = []
    for i, x in enumerate(lst):
        if i == 0 or x == lst[i - 1] + 1:
            this_lst.append(x)
        else:
            res.append(this_lst)
            this_lst = [x]

    # handle the last one
    if this_lst:
        res.append(this_lst)

    return res


def _horiz_note_onset_density(S: "MidiSongByMeasure",
                              tr_i: int,
                              measures: typing.Iterable[int],
                              denominator_includes_only_measures_with_note_ons=True) -> float or None:
    n_clicks = 0  # number of ticks for the denominator
    m_lens = S.get_measure_lengths()
    click_onsets = set()
    t = S.tracks[tr_i]
    for m_i in measures:
        t_m = t.tracks_by_measure[m_i]
        for n in t_m.note_ons:
            click_onsets.add((m_i, n.click))
        if t_m.note_ons or (not denominator_includes_only_measures_with_note_ons):
            n_clicks += m_lens[m_i]
    n_onsets = len(click_onsets)

    if n_clicks:
        return S.cpq * n_onsets / n_clicks
    else:
        return None


def _vert_note_onset_density(S: "MidiSongByMeasure",
                             tr_i: int,
                             measures: typing.Iterable[int]) -> float or None:
    """Assumes tr_i in S is sorted."""
    click_onsets_counter = collections.Counter()
    t = S.tracks[tr_i]
    for m_i in measures:
        t_m = t.tracks_by_measure[m_i]
        for n in t_m.note_ons:
            click_onsets_counter[(m_i, n.click)] += 1
    n_onsets = len(click_onsets_counter)
    if n_onsets:
        n_notes = sum(x for x in click_onsets_counter.values())
        return n_notes / n_onsets
    else:
        return None


def _vert_note_onset_n_pitch_classes_avg(S: MidiSongByMeasure,
                                         tr_i: int,
                                         measures: typing.Iterable[int]) -> float or None:
    """Assumes tr_i in S is sorted."""
    click_pitch_class_onsets = set()
    click_onsets = set()
    t = S.tracks[tr_i]
    for m_i in measures:
        t_m = t.tracks_by_measure[m_i]
        for n in t_m.note_ons:
            click_pitch_class_onsets.add((m_i, n.click, n.pitch % 12))
            click_onsets.add((m_i, n.click))
    if click_onsets:
        return len(click_pitch_class_onsets) / len(click_onsets)
    else:
        return None


def _pitch_interval_hist_for_contiguous_measure_slice(S: "MidiSongByMeasure",
                                                      tr_i: int,
                                                      measure_st: int,
                                                      measure_end: int) -> collections.Counter[float]:
    """Assumes tr_i in S is sorted."""
    L = S.get_measure_slice(st=measure_st, end=measure_end)
    hist = collections.Counter()

    def chord_dist(C_1, C_2):
        d = 0
        for n in C_1:
            smallest = min(abs(n.pitch - n_2.pitch) for n_2 in C_2)
            d += smallest
        return d / len(C_1)

    chords_in_order = []
    for m in L:
        t = m[tr_i]
        cur_click = -1
        cur_chord = []
        for note in t.note_ons:
            if note.click == cur_click:
                cur_chord.append(note)
            else:
                cur_click = note.click
                if cur_chord:
                    chords_in_order.append(cur_chord)
                cur_chord = [note]
        # handle last chord in this measure
        if cur_chord:
            chords_in_order.append(cur_chord)

    for i, c_2 in enumerate(chords_in_order):
        if i:
            c_1 = chords_in_order[i - 1]
            hist[chord_dist(c_1, c_2)] += 1

    return hist


def _pitch_interval_hist(S: MidiSongByMeasure,
                         tr_i: int,
                         measures: typing.Iterable[int]) -> dict[float, int]:
    """assumes tr_i in S is sorted"""
    L = _set_to_list_of_contiguous_sub_lists(measures)
    res = collections.Counter()
    for measures in L:
        d = _pitch_interval_hist_for_contiguous_measure_slice(S=S, tr_i=tr_i,
                                                              measure_st=measures[0],
                                                              measure_end=measures[-1] + 1)
        res.update(d)  # d is a collections.Counter object so this automatically sums values

    return dict(res)


def _consolidated_pitch_interval_hist(S: MidiSongByMeasure,
                                      tr_i: int,
                                      measures: typing.Iterable[int]
                                      ) -> "dict[str, float]":
    """assumes tr_i in S is sorted"""
    hist = _pitch_interval_hist(S=S, tr_i=tr_i, measures=measures)
    eps = 0.0001
    res = {'rep': 0, 'step': 0, 'leap': 0}
    total = 0
    for k, v in hist.items():
        total += v
        if k < eps:  # essentially, if k == 0
            res['rep'] += v
        elif k < 2 + eps:
            res['step'] += v
        else:
            res['leap'] += v

    res = dict(res)
    if total:
        for k, v in res.items():
            res[k] = v / total

    return res


# def _nice_pstdev(L: list, xbar=None):
#     if len(L) in (0, 1):
#         return 0.0
#     return statistics.pstdev(L, mu=xbar)


def _note_onset_click_differences_for_contiguous_measure_slice(S: MidiSongByMeasure,
                                                               tr_i: int,
                                                               measure_st: int,
                                                               measure_end: int) -> list[int]:
    """Assumes tr_i in S is sorted."""
    L = S.get_measure_slice(st=measure_st, end=measure_end)
    MEs = S.get_measure_endpoints(make_copy=False)
    note_onset_clicks = set()
    for m_i, m in enumerate(L):
        m_i = m_i + measure_st
        t = m[tr_i]
        this_measure_st_click = MEs[m_i]
        for note in t.note_ons:
            note_onset_clicks.add(note.click + this_measure_st_click)

    note_onset_clicks.add(MEs[measure_end])  # add fake note to the list, for the start of the next measure

    note_onset_clicks = list(note_onset_clicks)
    note_onset_clicks.sort()
    note_onset_differences = []
    for c_i, c in enumerate(note_onset_clicks):
        if c_i:
            note_onset_differences.append(c - note_onset_clicks[c_i - 1])
    return note_onset_differences


def _note_onset_vector_for_contiguous_measure_slice(S: MidiSongByMeasure,
                                                    tr_i: int,
                                                    measure_st: int,
                                                    measure_end: int) -> list[int]:
    L = S.get_measure_slice(st=measure_st, end=measure_end)
    MEs = S.get_measure_endpoints(make_copy=False)
    st_click = MEs[measure_st]
    end_click = MEs[measure_end] - 1  # inclusive
    res = [0] * (end_click - st_click + 1)
    for m_i, m in enumerate(L):
        m_i = m_i + measure_st
        t = m[tr_i]
        this_measure_st_click = MEs[m_i]
        for note in t.note_ons:
            click = note.click + this_measure_st_click
            res[click - st_click] = 1
    return res


def _horiz_note_onset_irregularity(S: "MidiSongByMeasure",
                                   tr_i: int,
                                   measures: typing.Iterable[int]) -> float:
    """assumes tr_i in S is sorted. Only uses the measures from 'measures' which contain note ons."""
    actual_measures = []
    t = S.tracks[tr_i]
    for m_i in measures:
        t_m = t.tracks_by_measure[m_i]
        if t_m.note_ons:
            actual_measures.append(m_i)
    measures = actual_measures

    L = _set_to_list_of_contiguous_sub_lists(measures)
    note_onset_differences = []
    for measures in L:
        x = _note_onset_click_differences_for_contiguous_measure_slice(S=S, tr_i=tr_i,
                                                                       measure_st=measures[0],
                                                                       measure_end=measures[-1] + 1)
        note_onset_differences.extend(x)

    note_onset_vectors = []
    for measures in L:
        v = _note_onset_vector_for_contiguous_measure_slice(S=S, tr_i=tr_i, measure_st=measures[0],
                                                            measure_end=measures[-1] + 1)
        note_onset_vectors.append(v)

    if len(note_onset_differences) > 1:
        # score_1 = 0
        # for d_i, d in enumerate(note_onset_differences):
        #     if d_i and note_onset_differences[d_i - 1] != d:
        #         score_1 += 1
        # score_1 = score_1 / (len(note_onset_differences) - 1)  # 0.0 <= score_1 <= 1.0

        # percentage of differences that are unique
        # score_2 = (len(set(note_onset_differences)) - 1) / (len(note_onset_differences) - 1)  # 0.0 <= score_2 <= 1.0

        # mu = statistics.mean(note_onset_differences) if note_onset_differences else 1.0
        # score_3 = _nice_pstdev(note_onset_differences, xbar=mu) / mu  # i.e. coefficient of variation

        score_4 = 1 - statistics.mean(_max_autocorr(v) for v in note_onset_vectors)

        # print(f'scores: {score_1} {score_2} {score_3} {score_4}')
        return score_4

    else:
        return 0.0


# def _horiz_note_onset_by_measure_stdev(S: "MidiSongByMeasure",
#                                        tr_i: int,
#                                        measures: typing.Iterable[int]) -> float or None:
#     densities = []
#     for m_i in measures:
#         tr = S.tracks[tr_i]
#         if tr.tracks_by_measure[m_i].note_ons:
#             this_density = S.horiz_note_onset_density(tr_i=tr_i, measures=[m_i])
#             densities.append(this_density)
#     # print('note onset densities', densities)
#     try:
#         xbar = statistics.mean(densities)
#         return statistics.stdev(densities, xbar=xbar)
#     except statistics.StatisticsError:
#         return None


def _horiz_note_onset_irregularity_new_idea(S: "MidiSongByMeasure",
                                   tr_i: int,
                                   measures: typing.Iterable[int],
                                   use_np=False) -> float:
    """assumes tr_i in S is sorted. Only uses the measures from 'measures' which contain note ons."""
    actual_measures = []
    t = S.tracks[tr_i]
    for m_i in measures:
        t_m = t.tracks_by_measure[m_i]
        if t_m.note_ons:
            actual_measures.append(m_i)
    measures = actual_measures

    # L = _set_to_list_of_contiguous_sub_lists(measures)
    #
    # note_onset_vectors = []
    # for measures in L:
    #     v = _note_onset_vector_for_contiguous_measure_slice(S=S, tr_i=tr_i, measure_st=measures[0],
    #                                                         measure_end=measures[-1] + 1)
    #     note_onset_vectors.append(v)

    note_onset_vectors = []
    for measure in measures:
        v = _note_onset_vector_for_contiguous_measure_slice(S=S, tr_i=tr_i, measure_st=measure, measure_end=measure + 1)
        note_onset_vectors.append(v)

    scores = []
    for v in note_onset_vectors:
        c = cyclic_autocorr(v, use_np=use_np)
        scores.append(len(set(c)))

    if scores:
        return statistics.mean(scores)
    else:
        return 0


def _max_autocorr(L):
    """L a binary vector (list)"""
    if len(L) <= 1:
        return 1.0
    if sum(L) == 1:
        return 1.0

    import numpy as np
    L = np.array(L, dtype='float64')
    mean = np.mean(L)
    L -= mean
    r = np.correlate(L, L, mode='full')
    c = (r[r.size//2:] / r[r.size//2])[1:]  # exclude maximum, which is the 0-shift value
    return float(max(abs(c)))  # abs is probably unnecessary for our musical examples


def _autocorr_naive(L):
    """L a binary vector (list)"""
    if len(L) <= 1:
        return 1.0
    if sum(L) == 1:
        return 1.0

    mean = statistics.mean(L)
    L = [x - mean for x in L]
    shifted_dots = []
    for shift in range(1, len(L)):
        shifted_L = [0] * shift
        for shift_2 in range(len(L) - shift):
            shifted_L.append(L[shift_2])
        dot = sum(x * y for x, y in zip(L, shifted_L))
        shifted_dots.append(dot)
    max_dot = sum(x * x for x in L)
    normalized_shifted_dots = [x / max_dot for x in shifted_dots]
    abs_applied = [abs(x) for x in normalized_shifted_dots]
    return float(max(abs_applied))


def cyclic_autocorr_np(x):
    """x a vector (list)"""
    import numpy as np
    return np.correlate(x, np.hstack((x[1:], x)), mode='valid')


def cyclic_autocorr_naive(L):
    """L a binary vector (list)"""
    res = []
    for i in range(len(L)):
        L_shifted = L[i:] + L[:i]
        L_dot = sum(a * b for a, b in zip(L, L_shifted))
        res.append(L_dot)
    return res


def cyclic_autocorr(x, use_np=False):
    if use_np:
        return cyclic_autocorr_np(x)
    else:
        return cyclic_autocorr_naive(x)
