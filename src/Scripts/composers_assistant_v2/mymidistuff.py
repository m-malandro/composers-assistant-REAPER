import collections
import typing
from reaper_python import *
import myfunctions as mf
import math
from collections import defaultdict


class MyNote(object):
    def __init__(self, take, noteidx, selected, muted, startppqpos, endppqpos, chan, pitch, vel):
        self.take = take
        self.noteidx = noteidx
        self.selected = selected
        self.muted = muted
        self.startppqpos = startppqpos
        self.endppqpos = endppqpos
        # self._startQN = RPR_MIDI_GetProjQNFromPPQPos(take, startppqpos)
        # self._endQN = RPR_MIDI_GetProjQNFromPPQPos(take, endppqpos)
        self.chan = chan
        self.pitch = pitch
        self.vel = vel
        self.__hash = hash(str(self))

    @property
    def startppqpos(self):  # getter
        return self._startppqpos

    @startppqpos.setter
    def startppqpos(self, val):  # setter
        self._startppqpos = int(round(val))
        self._startQN = RPR_MIDI_GetProjQNFromPPQPos(self.take, val)

    @property
    def endppqpos(self):  # getter
        return self._endppqpos

    @endppqpos.setter
    def endppqpos(self, val):  # setter
        self._endppqpos = int(round(val))
        self._endQN = RPR_MIDI_GetProjQNFromPPQPos(self.take, val)

    @property
    def startQN(self):  # getter
        return self._startQN

    @startQN.setter
    def startQN(self, val):  # setter
        self.startppqpos = int(RPR_MIDI_GetPPQPosFromProjQN(self.take, val))  # also sets ppqpos

    @property
    def endQN(self):  # getter
        return self._endQN

    @endQN.setter
    def endQN(self, val):  # setter
        self.endppqpos = int(RPR_MIDI_GetPPQPosFromProjQN(self.take, val))  # also sets ppqpos

    def __repr__(self):
        # return 'Note:' + ','.join(mf.tostr([self.noteidx, self.pitch, self.vel, self.chan, self.startQN, self.endQN, self.startppqpos, self.endppqpos, self.selected, self.muted]))
        return 'N:{}, vel:{}, chan:{}, startQN:{}, endQN:{}, startppq:{}, endppq:{}, sel:{}, muted:{}'.format(
            self.pitch, self.vel, self.chan, self.startQN, self.endQN, self.startppqpos, self.endppqpos, self.selected, self.muted)

    def __hash__(self):
        return self.__hash

    def __copy__(self):
        return MyNote(take=self.take, noteidx=self.noteidx, selected=self.selected, muted=self.muted,
                      startppqpos=self.startppqpos, endppqpos=self.endppqpos, chan=self.chan, pitch=self.pitch,
                      vel=self.vel)


def get_notes(take, sel_only=False, QN_window=None, unmuted_only=False) -> "dict[int, MyNote]":
    """QN_window None (no restriction) or, e.g., [4, 13] to get all notes with 4<=start time<13. Returns a dict
    idx:my_note obj """
    res = {}
    retval, take, notecntOut, ccevtcntOut, textsyxevtcntOut = RPR_MIDI_CountEvts(take, 0, 0, 0)
    for idx in range(notecntOut):
        retval, take, noteidx, selectedOut, mutedOut, startppqposOut, endppqposOut, chanOut, pitchOut, velOut = RPR_MIDI_GetNote(
                  take, idx, 0, 0, 0, 0, 0, 0, 0)
        if (not sel_only) or (selectedOut):
            startQN = RPR_MIDI_GetProjQNFromPPQPos(take, startppqposOut)
            if (QN_window is None) or (QN_window[0] <= startQN < QN_window[1]):
                if not (unmuted_only and mutedOut):
                    res[noteidx] = MyNote(take, noteidx, selectedOut, mutedOut, startppqposOut, endppqposOut, chanOut, pitchOut, velOut)
    return res


def get_sel_notes(take):
    res = {}

    retval, take, notecntOut, ccevtcntOut, textsyxevtcntOut = RPR_MIDI_CountEvts(take, 0, 0, 0)  # count notes
    if notecntOut == 0:  # if no notes, there can't be any selected notes
        return res

    noteidx = RPR_MIDI_EnumSelNotes(take, -1)
    while noteidx != -1:
        retval, take, noteidx, selectedOut, mutedOut, startppqposOut, endppqposOut, chanOut, pitchOut, velOut = RPR_MIDI_GetNote(
            take, noteidx, 0, 0, 0, 0, 0, 0, 0)
        res[noteidx] = MyNote(take, noteidx, selectedOut, mutedOut, startppqposOut, endppqposOut, chanOut, pitchOut, velOut)
        noteidx = RPR_MIDI_EnumSelNotes(take, noteidx)

    return res


def correct_note_overlaps(d):
    """In place operation. d a dict of the form idx:MyNote object, e.g., d an output of get_notes

    :type d: dict[int, MyNote]"""
    by_pitch = collections.defaultdict(list)
    for idx, n in d.items():
        by_pitch[n.pitch].append(n)
    for pitch, L in by_pitch.items():
        L.sort(key=lambda x: x.startppqpos)
    for pitch, L in by_pitch.items():
        for i, n in enumerate(L):
            if i:
                prev = L[i - 1]
                if prev.endppqpos > n.startppqpos:
                    prev.endppqpos = n.startppqpos


def insert_MyNotes(take, notes, noSortInOptional):
    """notes a list of MyNote objects"""
    for note in notes:
        RPR_MIDI_InsertNote(take, note.selected, note.muted, note.startppqpos, note.endppqpos, note.chan, note.pitch,
                            note.vel, noSortInOptional)


def delete_notes_by_idx(take, L):
    """L a list of noteidx's"""
    for i, noteidx in enumerate(L):
        RPR_MIDI_DeleteNote(take, noteidx - i)


def write_notes_to_take(take, note_idxs_to_del: typing.Iterable[int], notes_to_write: typing.Dict[int, MyNote], noSortInOptional=True):
    correct_note_overlaps(d=notes_to_write)
    delete_notes_by_idx(take=take, L=note_idxs_to_del)
    to_write = list(notes_to_write.values())
    insert_MyNotes(take=take, notes=to_write, noSortInOptional=noSortInOptional)


def set_note_properties(take, MyNote_instance, set_selected=None, set_muted=None,
                        set_startppqpos=None, set_endppqpos=None, set_chan=None, set_pitch=None,
                        set_vel=None, noSortInOptional=False):
    note = MyNote_instance

    if set_selected is None:
        selected = -1
    else:
        selected = bool(set_selected)

    if set_muted is None:
        muted = note.muted  # b/c of bug in REAPER
    else:
        muted = bool(set_muted)

    if set_startppqpos is None:
        startppqpos = -1
    else:
        startppqpos = int(set_startppqpos)

    if set_endppqpos is None:
        endppqpos = -1
    else:
        endppqpos = int(set_endppqpos)

    # some correction for erroneous inputs
    # (if the user tries to set startppqpos after the note end or endppqpos before the note beginning)
    if startppqpos > -1 and endppqpos > -1:  # if we're setting both the startppqpos and the endppqpos
        endppqpos = max(startppqpos, endppqpos)
    elif startppqpos > -1 and endppqpos == -1:  # if we're setting the startppqpos but not the endppqpos
        if startppqpos > note.endppqpos:
            endppqpos = startppqpos
    elif startppqpos == -1 and endppqpos > -1:  # if we're setting the endppqpos but not the startppqpos
        if note.startppqpos > endppqpos:
            startppqpos = endppqpos

    if set_chan is None:
        chan = -1
    else:
        chan = int(set_chan)

    if set_pitch is None:
        pitch = -1
    else:
        pitch = int(set_pitch)

    if set_vel is None:
        vel = -1
    else:
        vel = int(set_vel)

    RPR_MIDI_SetNote(take, note.noteidx, selected, muted, startppqpos, endppqpos, chan, pitch, vel, noSortInOptional)


class MyChord(object):
    def __init__(self, notes=None):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

        self.compute_attributes()

    def compute_attributes(self):

        pitches = [note.pitch for note in self.notes]
        pitches.sort()
        self.notes.sort(key=lambda x: x.pitch)

        self.pitches = pitches
        self.pitch_center_chord_idx = math.floor((len(pitches)-1)/2.0)  # we choose to round down
        self.pitch_center_note_idx = self.notes[self.pitch_center_chord_idx].noteidx
        self.center_qn = self.notes[self.pitch_center_chord_idx].startQN
        self.pitch_center = self.notes[self.pitch_center_chord_idx].pitch

        self.notes.sort(key=lambda x: x.noteidx)

    def add_note(self, note):
        self.notes.append(note)
        self.compute_attributes()

    def __iter__(self):
        for note in self.notes:
            yield note


class MyCC(object):
    def __init__(self, take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out):
        self.take = take
        self.ccidx = ccidx
        self.selected = selectedOut
        self.muted = mutedOut
        self.ppqpos = ppqposOut
        self.chanmsg = chanmsgOut
        self.chan = chanOut
        self.msg2 = msg2Out
        self.msg3 = msg3Out
        # self.__hash = hash(str(self))
        # add hash? only if needed

    def __repr__(self):
        return 'CC:' + ','.join(mf.tostr([self.ccidx, self.selected, self.muted, self.ppqpos, self.chanmsg, self.chan, self.msg2, self.msg3]))


def get_CCs(take, sel_only=False):
    res = {}
    retval, take, notecntOut, ccevtcntOut, textsyxevtcntOut = RPR_MIDI_CountEvts(take, 0, 0, 0)
    for idx in range(ccevtcntOut):
        retval, take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out = RPR_MIDI_GetCC(
            take, idx, 0, 0, 0, 0, 0, 0, 0)
        if (not sel_only) or selectedOut:
            res[ccidx] = MyCC(take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out)
    return res


def CC_iter(take, sel_only=False):
    retval, take, notecntOut, ccevtcntOut, textsyxevtcntOut = RPR_MIDI_CountEvts(take, 0, 0, 0)
    for idx in range(ccevtcntOut):
        retval, take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out = RPR_MIDI_GetCC(
            take, idx, 0, 0, 0, 0, 0, 0, 0)
        if (not sel_only) or selectedOut:
            yield MyCC(take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out)


def get_sel_CCs(take):
    res = {}
    ccidx=-1
    ccidx = RPR_MIDI_EnumSelCC(take, ccidx)
    while ccidx != -1:
        retval, take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out = RPR_MIDI_GetCC(
            take, ccidx, 0, 0, 0, 0, 0, 0, 0)
        res[ccidx] = MyCC(take, ccidx, selectedOut, mutedOut, ppqposOut, chanmsgOut, chanOut, msg2Out, msg3Out)
        ccidx = RPR_MIDI_EnumSelCC(take, ccidx)
    return res


# good
def next_grid_point_QN(x_in_QN, grid_QN, grid_swing, tol = 0.001):
    """returns the next grid point, in quarter note value, to x, where x is given in quarter note value."""
    x = math.floor(x_in_QN/grid_QN)
    if x % 2 == 1:
        x = x-1

    swing_pos = x*grid_QN + grid_QN*(0.5*grid_swing+1)

    if x_in_QN < swing_pos and abs(x_in_QN-swing_pos) > tol:
        closest = swing_pos
    else:
        closest = (x+2)*grid_QN
    # closest is in QN right now
    return closest


# good
def previous_grid_point_QN(x_in_QN, grid_QN, grid_swing, tol = 0.001):
    """returns the previous grid point, in quarter note value, to x, where x is given in quarter note value."""
    x = math.floor(x_in_QN / grid_QN)
    if x % 2 == 1:
        x = x-1

    swing_pos = x*grid_QN + grid_QN*(0.5*grid_swing+1)

    # REAPER appears to snap notes to just after grid points if it cannot hit the grid exactly
    if x_in_QN > swing_pos and abs(x_in_QN - swing_pos) > tol:
        closest = swing_pos
    else:
        closest = x*grid_QN
    # closest is in QN right now
    return closest


# good
def closest_grid_point_QN(x_in_QN, grid_QN, grid_swing):
    """returns the closest grid point, in quarter note value, to x, where x is given in quarter note value."""
    x = math.floor(x_in_QN / grid_QN)
    if x % 2 == 1:
        x = x-1

    swing_pos = x*grid_QN + grid_QN*(0.5*grid_swing+1)

    closest = x*grid_QN
    if abs(x_in_QN - swing_pos) < abs(x_in_QN - closest):
        closest = swing_pos
    if abs(x_in_QN - (x+2)*grid_QN) < abs(x_in_QN - closest):
        closest = (x+2)*grid_QN
    # closest is in QN now
    return closest


def closest_grid_point_ppq(x_in_ppq, take, grid_QN, grid_swing):
    """returns the closest grid point, in ppq value, to x, where x is given in ppq value"""
    x_in_QN = ppq_to_QN(x_in_ppq, take)
    res_in_QN = closest_grid_point_QN(x_in_QN=x_in_QN, grid_QN=grid_QN, grid_swing=grid_swing)
    res_in_ppq = QN_to_ppq(x_in_QN=res_in_QN, take=take)
    return res_in_ppq


def previous_grid_point_ppq(x_in_ppq, take, grid_QN, grid_swing, tol=0.001):
    """returns the previous grid point, in ppq value, to x, where x is given in ppq value."""
    x_in_QN = ppq_to_QN(x_in_ppq=x_in_ppq, take=take)
    res_in_QN = previous_grid_point_QN(x_in_QN=x_in_QN, grid_QN=grid_QN, grid_swing=grid_swing, tol=tol)
    res_in_ppq = QN_to_ppq(x_in_QN=res_in_QN, take=take)
    return res_in_ppq


def next_grid_point_ppq(x_in_ppq, take, grid_QN, grid_swing, tol=0.001):
    """returns the next grid point, in ppq value, to x, where x is given in ppw value."""
    x_in_QN = ppq_to_QN(x_in_ppq=x_in_ppq, take=take)
    res_in_QN = next_grid_point_QN(x_in_QN=x_in_QN, grid_QN=grid_QN, grid_swing=grid_swing, tol=tol)
    res_in_ppq = QN_to_ppq(x_in_QN=res_in_QN, take=take)
    return res_in_ppq


def get_grid(take):
    """returns (grid, swing) values for the most recent MIDI editor for this take, in quarter notes. Swing is in [-1,1].
    Examples:
        1/16 grid, straight gives (0.25, 0)
        1/16 grid, triplet gives (0.166666666..., 0)
        1/16 grid, dotted gives (0.375, 0)
        1/16 swing with 55% swing gives (0.25, 0.55)
    """
    retval, take, swingOutOptional, noteLenOutOptional = RPR_MIDI_GetGrid(take, 0, 0)
    return retval, swingOutOptional


def ppq_to_QN(x_in_ppq, take):
    """ Returns the project time in quarter notes corresponding to a specific MIDI tick (ppq) position."""
    return RPR_MIDI_GetProjQNFromPPQPos(take, x_in_ppq)


def QN_to_ppq(x_in_QN, take):
    """ Returns the MIDI tick (ppq) position corresponding to a specific project time in quarter notes."""
    return int(round(RPR_MIDI_GetPPQPosFromProjQN(take, x_in_QN)))


def earliest_note_in_list(L):
    """L a list of notes. Returns a note with earliest startppqpos"""
    min_note = None
    min_ppq = None
    for n in L:
        if min_note is None:
            min_note = n
            min_ppq = n.startppqpos
        else:
            if n.startppqpos < min_ppq:
                min_note = n
                min_ppq = n.startppqpos

    if min_note is None or min_ppq is None:
        raise ValueError('input to earliest_note_in_list is empty')
    else:
        return min_note


def latest_note_in_list(L):
    """L a list of notes. Returns a note with latest startppqpos"""
    max_note = None
    max_ppq = None
    for n in L:
        if max_note is None:
            max_note = n
            max_ppq = n.startppqpos
        else:
            if n.startppqpos > max_ppq:
                max_note = n
                max_ppq = n.startppqpos

    if max_note is None or max_ppq is None:
        raise ValueError('input to latest_note_in_list is empty')
    else:
        return max_note


def get_track_names() -> list:
    res = []
    for trackidx in range(RPR_CountTracks(0)):
        track = RPR_GetTrack(0, trackidx)
        retval, track, flagsOut = RPR_GetTrackState(track, 0)
        track_name = retval
        res.append(track_name)
    return res


def get_tracks() -> list:
    res = []
    for trackidx in range(RPR_CountTracks(0)):
        track = RPR_GetTrack(0, trackidx)
        res.append(track)
    return res


class MeasureInfo(object):
    def __init__(self, measure_number, time_st_in_sec, qn_st, qn_end, time_sig_numerator, time_sig_denominator, tempo):
        self.measure_number = measure_number
        self.time_st_in_sec = time_st_in_sec
        self.qn_st = qn_st
        self.qn_end = qn_end
        self.time_sig_numerator = time_sig_numerator
        self.time_sig_denominator = time_sig_denominator
        self.tempo = tempo

    def __repr__(self):
        return 'Measure:' + ','.join(mf.tostr(
            [self.measure_number,
             self.qn_st, self.qn_end, self.time_st_in_sec, self.tempo, self.time_sig_numerator,
             self.time_sig_denominator]))


def get_measure_info(i):
    """returns information about measure i in current project"""
    RES = RPR_TimeMap_GetMeasureInfo(0, i, 0, 0, 0, 0, 0)
    retval, _, _, qn_start, qn_end, timesig_numerator, timesig_denominator, tempo = RES
    return MeasureInfo(measure_number=i, time_st_in_sec=retval, qn_st=qn_start, qn_end=qn_end,
                       time_sig_numerator=timesig_numerator,
                       time_sig_denominator=timesig_denominator, tempo=tempo)


def get_note_measures(take, selected_notes_only=True):
    """notes are assigned to measures based on their start times"""
    if selected_notes_only:
        get_function = get_sel_notes
    else:
        get_function = get_notes

    notes_by_idx = get_function(take=take)

    measure_to_list_of_notes = defaultdict(list)
    for i, note in notes_by_idx.items():
        retval, proj, qn, qnMeasureStart, qnMeasureEnd = RPR_TimeMap_QNToMeasures(0, note.startQN, 0, 0)
        retval = retval - 1  # TO COMPENSATE FOR REAPER RETURNING A 1-BASED INDEX.
        measure_to_list_of_notes[retval].append(note)

    return measure_to_list_of_notes


def QN_to_measure(QN):
    retval, proj, qn, qnMeasureStart, qnMeasureEnd = RPR_TimeMap_QNToMeasures(0, QN, 0, 0)
    retval = retval - 1  # to compensate for REAPER returning a 1-based index.
    return retval


def get_enclosing_measure_numbers(notes_by_index):
    if not notes_by_index:
        return []

    low = None
    hi = None
    for i, note in notes_by_index.items():
        if low is None:
            low = note.startQN
        if hi is None:
            hi = note.startQN

        if note.startQN < low:
            low = note.startQN
        if note.startQN > hi:
            hi = note.startQN

    start_measure = QN_to_measure(low)
    end_measure = QN_to_measure(hi)
    return [start_measure, end_measure+1]


def get_topmost_or_bottommost_notes(take, k, notes_by_idx, top_or_bottom='top'):
    """get the kth 'layer' of notes from the top or bottom. k=0 = top or bottom layer"""
    notes = notes_by_idx.values()
    if top_or_bottom == 'top':
        notes = sorted(notes, key=lambda n: -n.pitch)  # notes are in order, highest pitch = lowest index
    else:
        notes = sorted(notes, key=lambda n: n.pitch)  # notes are in order, lowest pitch = lowest index

    res = {}

    if notes:
        hi_pitch = notes[0].pitch
    else:
        return res

    def is_blocked(note, blocked_times):
        for T in blocked_times:
            # these functions should be modified to use approx instead of ==,
            # and should also allow the user to decide whether or not small overlaps "count"
            if min(T[1], note.endQN) - max(T[0], note.startQN) > 0:  # if these intervals overlap
                return True
            if note.startQN == note.endQN and T[0] <= note.startQN < T[1]:
                return True
            if T[0] == T[1] and note.startQN <= T[0] < note.endQN:
                return True
            if note.startQN == note.endQN and T[0] == T[1] and note.startQN == T[0]:
                return True
        return False

    blocked_times = []
    for note in notes:

        if note.pitch == hi_pitch or not is_blocked(note, blocked_times):
            res[note.noteidx] = note

            blocked_times.append((note.startQN, note.endQN))

    if k == 0:
        return res
    else:
        res2 = {}
        for i, note in notes_by_idx.items():
            if i not in res:
                res2[i] = note
        return get_topmost_or_bottommost_notes(take, k-1, res2, top_or_bottom)


def add_pitch_bend_at_time(take, t: float):
    if take:
        import random
        import mytrackviewstuff as mt
        qn = mt.time_to_QN(t)
        ppq = QN_to_ppq(qn, take)

        selected = 0  # bool
        muted = 0  # bool
        chanmsg = 224  # pitch bend, channel 1
        chan = 0  # redundant, but I guess it needs it

        msg2 = 0  # lsb, 0-255
        msg3 = 64  # msb, 0-255
        # insert pb event at 0 bend
        RPR_MIDI_InsertCC(take, selected, muted, ppq, chanmsg, chan, msg2, msg3)
        # then FIND the cc you just inserted and set its shape to linear
        for cc in CC_iter(take):
            if cc.msg2 == msg2 and cc.msg3 == msg3 and cc.chanmsg == chanmsg and cc.ppqpos == ppq:
                RPR_MIDI_SetCCShape(take, cc.ccidx, 5, -1.0, 1)
                break

        msg2 = 0
        msg3 = int(mt.get_ext_state('my_pitch_bend', 'pb_amount', '0'))
        ppq += 1
        # insert pb event at bend amount
        RPR_MIDI_InsertCC(take, selected, muted, ppq, chanmsg, chan, msg2, msg3)
        for cc in CC_iter(take):
            if cc.msg2 == msg2 and cc.msg3 == msg3 and cc.chanmsg == chanmsg and cc.ppqpos == ppq:
                RPR_MIDI_SetCCShape(take, cc.ccidx, 5, random.random() * 0.2, 1)
                break

        # insert pb event at 0 bend
        pb_length = float(mt.get_ext_state('my_pitch_bend', 'pb_len', '0.23'))
        ppq = QN_to_ppq(qn + pb_length, take)
        msg2 = 0
        msg3 = 64
        RPR_MIDI_InsertCC(take, selected, muted, ppq, chanmsg, chan, msg2, msg3)
        for cc in CC_iter(take):
            if cc.msg2 == msg2 and cc.msg3 == msg3 and cc.chanmsg == chanmsg and cc.ppqpos == ppq:
                RPR_MIDI_SetCCShape(take, cc.ccidx, 1, 0, 1)
                break


def MIDI_sort(take):
    RPR_MIDI_Sort(take)
