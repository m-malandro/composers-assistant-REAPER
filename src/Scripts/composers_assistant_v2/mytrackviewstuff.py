from reaper_python import *
import myfunctions as mf
import mymidistuff as mm
import collections

_WARNINGS_PRINTED = set()

class EnvelopePoint(object):
    def __init__(self, envelope, autoitem_idx, ptidx, time, value, shape, tension, selected):
        self.envelope = envelope
        self.autoitem_idx = autoitem_idx
        self.ptidx = ptidx
        self.time = time
        self.value = value
        self.shape = shape
        self.tension = tension
        self.selected = selected

    def __repr__(self):
        return 'EnvPt:' + ','.join(mf.tostr([self.autoitem_idx, self.ptidx, self.time, self.value, self.shape, self.tension, self.selected]))


def get_envelope_points(track_envelope, selected_only=False):
    """Returns a list L of lists. L[0] is the list of envelope points in the underlying envelope.
    L[1] is the list of envelope points in the first automation item on the envelope.
    L[2] is the list of envelope points in the second automation item on the envelope.
    etc."""
    c = RPR_CountEnvelopePoints(track_envelope)  # -1 if no envelope selected
    AI_counts = []
    for i in range(RPR_CountAutomationItems(track_envelope)):
        AI_counts.append(RPR_CountEnvelopePointsEx(track_envelope, i))

    res = []
    underlying_envelope_points = []
    for i in range(c):
        retval, _, autoitem_idx, ptidx, timeOut, valueOut, shapeOut, tensionOut, selectedOut = RPR_GetEnvelopePointEx(
            track_envelope, -1, i, -1, -1, -1, -1, -1)
        this_point = EnvelopePoint(envelope=track_envelope, autoitem_idx=autoitem_idx, ptidx=ptidx, time=timeOut,
                                   value=valueOut, shape=shapeOut, tension=tensionOut, selected=selectedOut)
        if selected_only:
            if this_point.selected:
                underlying_envelope_points.append(this_point)
        else:
            underlying_envelope_points.append(this_point)
    res.append(underlying_envelope_points)

    for ai_c_i, ai_c in enumerate(AI_counts):  # index, actual count
        this_AI_point_list = []
        for i in range(ai_c):
            retval, _, autoitem_idx, ptidx, timeOut, valueOut, shapeOut, tensionOut, selectedOut = RPR_GetEnvelopePointEx(
                track_envelope, ai_c_i, i, -1, -1, -1, -1, -1)
            this_point = EnvelopePoint(envelope=track_envelope, autoitem_idx=autoitem_idx, ptidx=ptidx, time=timeOut,
                                       value=valueOut, shape=shapeOut, tension=tensionOut, selected=selectedOut)
            if selected_only:
                if this_point.selected:
                    this_AI_point_list.append(this_point)
            else:
                this_AI_point_list.append(this_point)
        res.append(this_AI_point_list)

    return res


class TrackState(object):
    def __init__(self, trackidx, name, flags):
        self.trackidx = trackidx
        self.name = name

        int_flags = flags
        flags = bin(flags)

        self.is_folder = (int_flags >= 1 and flags[-1] == '1')
        self.is_selected = (int_flags >= 2 and flags[-2] == '1')
        self.has_fx_enabled = (int_flags >= 4 and flags[-3] == '1')
        self.is_muted = (int_flags >= 8 and flags[-4] == '1')
        self.is_soloed = (int_flags >= 16 and flags[-5] == '1')
        self.is_soloed_in_place = (int_flags >= 32 and flags[-5] == '1' and flags[-6] == '1')
        self.is_rec_armed = (int_flags >= 64 and flags[-7] == '1')
        self.rec_monitoring_on = (int_flags >= 128 and flags[-8] == '1')
        self.rec_monitoring_auto = (int_flags >= 256 and flags[-9] == '1')
        self.hide_from_TCP = (int_flags >= 512 and flags[-10] == '1')
        self.hide_from_MCP = (int_flags >= 1024 and flags[-11] == '1')

        self.flags = flags
        self.int_flags = int_flags

    def __repr__(self):
        res = "Track {}".format(self.trackidx)
        if self.name:
            res += ': ' + self.name
        if self.is_folder:
            res += '; folder'
        if self.is_selected:
            res += '; sel'
        if self.has_fx_enabled:
            res += '; fx on'
        if self.is_muted:
            res += '; muted'
        if self.is_soloed:
            res += '; solo'
        if self.is_soloed_in_place:
            res += '; SIP'
        if self.is_rec_armed:
            res += '; rec armed'
        if self.rec_monitoring_on:
            res += '; rec monitoring on'
        if self.rec_monitoring_auto:
            res += '; rec monitoring auto'
        if self.hide_from_TCP:
            res += '; hide from TCP'
        if self.hide_from_MCP:
            res += '; hide from MCP'

        return res


def get_tracks_by_idx():
    """returns a dict of the form int: track object"""
    res = {}
    for trackidx in range(RPR_CountTracks(0)):
        track = RPR_GetTrack(0, trackidx)
        res[trackidx] = track
    return res


def get_sel_tracks_by_idx():
    """returns a dict of the form int: track object"""
    res = {}
    tracks_by_idx = get_tracks_by_idx()
    for i, track in tracks_by_idx.items():
        if get_track_state(i).is_selected:
            res[i] = track
    return res


def get_track_state(trackidx):
    """returns a TrackState object for the input track object"""
    track = RPR_GetTrack(0, trackidx)
    retval, track, flagsOut = RPR_GetTrackState(track, 0)
    return TrackState(trackidx=trackidx, name=retval, flags=flagsOut)


def get_num_FX_on_track(track):
    return RPR_TrackFX_GetCount(track)


def get_FX_names_on_track(track, max_chars=1000):
    res = []
    for i in range(get_num_FX_on_track(track)):
        retval, track, fx, bufOut, bufOut_sz = RPR_TrackFX_GetFXName(track, i, "", max_chars)
        res.append(bufOut)
    return res


def rename_fx(trackidx, fx_number, new_name):
    command_int = RPR_NamedCommandLookup("_RSb5f482456875c21f87f804f4e43d0c703eb7918e")  # SetFXName_CallFromExt.lua
    RPR_SetExtState("SetFXName", "trackidx", trackidx, False)
    RPR_SetExtState("SetFXName", "fx_number", fx_number, False)
    RPR_SetExtState("SetFXName", "new_name", new_name, False)
    RPR_Main_OnCommand(command_int, 0)


def sec_to_QN(sec):
    return RPR_TimeMap_timeToQN(sec)


def QN_to_sec(qn):
    return RPR_TimeMap_QNToTime(qn)


class TakeWithInfo(object):
    def __init__(self, take, track_idx):
        self.take = take

        # parent (MediaItem) related stuff
        self.parent = RPR_GetMediaItemTake_Item(take)
        self.track = RPR_GetMediaItemTrack(self.parent)
        self.track_idx = track_idx

        self.start_sec = RPR_GetMediaItemInfo_Value(self.parent, "D_POSITION")
        self.start_QN = sec_to_QN(self.start_sec)
        self.length_sec_in_track = RPR_GetMediaItemInfo_Value(self.parent, "D_LENGTH")
        self.end_sec_in_track = self.start_sec + self.length_sec_in_track
        self.end_QN_in_track = sec_to_QN(self.end_sec_in_track)
        self.length_QN_in_track = self.end_QN_in_track - self.start_QN
        self.snap_offset = RPR_GetMediaItemInfo_Value(self.parent, "D_SNAPOFFSET")
        self.is_looped = RPR_GetMediaItemInfo_Value(self.parent, "B_LOOPSRC")
        self.is_muted = RPR_GetMediaItemInfo_Value(self.parent, "B_MUTE")
        self.is_selected = RPR_IsMediaItemSelected(self.parent)

        # take stuff
        self.start_offset_sec = RPR_GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")  # only relevant if looped

        src = RPR_GetMediaItemTake_Source(take)
        src_type = RPR_GetMediaSourceType(src, '', 1000)
        self.source_type = src_type[1]

        retval, _, is_QN_length = RPR_GetMediaSourceLength(src, 1)
        if not is_QN_length:
            retval = sec_to_QN(retval)
        self.source_length_QN = retval


def get_active_takes_with_info_by_trackidx(include_muted_tracks=False, midi_items_only=True, include_muted_items=False,
                                           excl_track_name_substrings=None) -> "dict[int, list[TakeWithInfo]]":
    res = collections.defaultdict(list)
    if excl_track_name_substrings is None:
        excl_track_name_substrings = []

    for track_i, track in get_tracks_by_idx().items():

        if include_muted_tracks or not get_track_state(track_i).is_muted:

            track_name, _, _ = RPR_GetTrackState(track, 0)
            if all([x not in track_name for x in excl_track_name_substrings]):

                for media_item_i in range(RPR_GetTrackNumMediaItems(track)):
                    media_item = RPR_GetTrackMediaItem(track, media_item_i)
                    take = RPR_GetActiveTake(media_item)
                    candidate = TakeWithInfo(take=take, track_idx=track_i)
                    if candidate.source_type in ('MIDI', 'MIDIPOOL') or not midi_items_only:
                        if include_muted_items or not candidate.is_muted:
                            res[track_i].append(candidate)

    return res


def get_displayed_notes_by_idx_in_take_with_info(take_with_info, qn_window_st=0, qn_window_end=9999999,
                                                 display_warnings=True):

    if take_with_info.is_looped and display_warnings:
        warning = 'WARNING: Looped midi items are currently unsupported.'
        warning += ' If you are unhappy with the outcome of this function, please un-loop all midi items and try again.'
        if warning not in _WARNINGS_PRINTED:
            print(warning)
        _WARNINGS_PRINTED.add(warning)

    notes_by_idx = mm.get_notes(take_with_info.take,
                                QN_window=(qn_window_st, qn_window_end),
                                unmuted_only=True
                                )

    # now truncate note ends based on take_w_info endpoint
    for i, n in notes_by_idx.items():
        if n.endQN > take_with_info.end_QN_in_track:
            n.endQN = take_with_info.end_QN_in_track

    return notes_by_idx


def get_time_selection():
    """Returns None or a time tuple"""
    is_set, is_loop, start, end, allowautoseek = RPR_GetSet_LoopTimeRange(0, 0, 0, 0, 0)
    if mf.is_approx(start-end, 0):
        return None
    else:
        return start, end


def get_all_visible_midi_notes_by_trackidx(attach_parent_take_to_each_note=False,
                                           include_muted_items=False,
                                           include_muted_tracks=False,
                                           restrict_to_time_selection=True,
                                           excl_track_name_substrings=None,
                                           display_warnings=True):
    if restrict_to_time_selection:
        time_selection = get_time_selection()
        if time_selection is None:
            restrict_to_time_selection = False
        else:
            time_selection_QN = (sec_to_QN(time_selection[0]), sec_to_QN(time_selection[1]))

    AT = get_active_takes_with_info_by_trackidx(include_muted_tracks=include_muted_tracks,
                                                include_muted_items=include_muted_items,
                                                excl_track_name_substrings=excl_track_name_substrings)
    res = {}
    for i, TI in AT.items():
        res[i] = []
        for take_with_info in TI:
            if include_muted_items or not take_with_info.is_muted:
                qn_window_st = take_with_info.start_QN-.0001
                qn_window_end = take_with_info.end_QN_in_track-.0001
                if restrict_to_time_selection:
                    qn_window_st = max(qn_window_st, time_selection_QN[0]-.0001)
                    qn_window_end = min(qn_window_end, time_selection_QN[1]-.0001)

                d = get_displayed_notes_by_idx_in_take_with_info(take_with_info=take_with_info,
                                                                 qn_window_st=qn_window_st,
                                                                 qn_window_end=qn_window_end,
                                                                 display_warnings=display_warnings)
                for _, note in d.items():
                    if attach_parent_take_to_each_note:
                        note.parent_take = take_with_info.take
                    res[i].append(note)
    return res


def clear_unmuted_visible_notes_in_time_selection(display_warnings=True):
    notes_by_trackidx = get_all_visible_midi_notes_by_trackidx(attach_parent_take_to_each_note=True,
                                                               restrict_to_time_selection=True,
                                                               display_warnings=display_warnings)
    takes = {}
    for track_idx, note_L in notes_by_trackidx.items():
        for n in note_L:
            if n.take not in takes:
                takes[n.take] = []
            takes[n.take].append(n.noteidx)

    for take in takes:
        mm.delete_notes_by_idx(take, takes[take])


def get_measure_endpoints_in_QN(first_measure_index=0, last_measure_index=1):
    """indexes are inclusive. Gets all endpoints from the start of the first measure to the end of the last."""

    res = []

    for mi in range(first_measure_index, last_measure_index+2):
        RES = RPR_TimeMap_GetMeasureInfo(0, mi, 0, 0, 0, 0, 0)
        retval, _, _, qn_start, qn_end, timesig_numerator, timesig_denominator, tempo = RES

        res.append(qn_start)

    return res


def qn_to_measure(qn):
    retval, proj, qn, _, _ = RPR_TimeMap_QNToMeasures(0, qn, 0, 0)
    return retval - 1  # to compensate for Reaper returning a 1-based index.


# class TimeSigTempoMarker(object):
#     def __init__(self, idx, timepos, measure, beatpos, num, denom, linear_tempo):
#         self.idx = idx
#         self.timepos = timepos
#         self.measure = measure
#         self.beatpos = beatpos
#         self.num = num
#         self.denom = denom
#         self.linear_tempo = linear_tempo
#
#
# def get_master_tempo() -> float:
#     return RPR_Master_GetTempo()
#
#
# def get_tempo_markers():
#     res = []
#
#     has_tempo_markers = False
#     for i in range(RPR_CountTempoTimeSigMarkers(0)):
#         has_tempo_markers = True
#         retval, proj, ptidx, timeposOut, measureposOut, beatposOut, bpmOut, timesig_numOut, timesig_denomOut, lineartempoOut = RPR_GetTempoTimeSigMarker(
#             0, i, 0, 0, 0, 0, 0, 0, 0)
#
#         res.append(TimeSigTempoMarker(idx=ptidx, timepos=timeposOut, measure=measureposOut, beatpos=beatposOut,
#                                       num=timesig_numOut, denom=timesig_denomOut, linear_tempo=lineartempoOut))
#
#     if not has_tempo_markers:
#         res.append(TimeSigTempoMarker(idx=-1, timepos=0, measure=0, beatpos=0, num=???))
#
#     return res

def get_time_sig_at_st_of_measure(measure_idx):
    retval, proj, measure, qn_startOut, qn_endOut, timesig_numOut, timesig_denomOut, tempoOut = RPR_TimeMap_GetMeasureInfo(
        0, measure_idx, 0, 0, 0, 0, 0)
    return timesig_numOut, timesig_denomOut


def get_tempo_at_st_of_measure(measure_idx):
    retval, proj, measure, qn_startOut, qn_endOut, timesig_numOut, timesig_denomOut, tempoOut = RPR_TimeMap_GetMeasureInfo(
        0, measure_idx, 0, 0, 0, 0, 0)
    return tempoOut


def messagebox(msg, title, int_type):
    """int_type: 0=OK, 1=OKCANCEL, 2=ABORTRETRYIGNORE, 3=YESNOCANCEL, 4=YESNO, 5=RETRYCANCEL:
    ret 1=OK, 2=CANCEL, 3=ABORT, 4=RETRY, 5=IGNORE, 6=YES, 7=NO"""
    RPR_ShowMessageBox(msg, title, int_type)


def is_SWS_installed() -> bool:
    if RPR_APIExists("CF_GetSWSVersion"):
        return True
    return False


def get_ext_state(section: str, key: str, default_value: str = "", return_default_if_doesnt_exist=True) -> str:
    if RPR_HasExtState(section, key):
        return RPR_GetExtState(section, key)

    if return_default_if_doesnt_exist:
        return default_value

    raise KeyError(f"key={key} doesn't exist in section={section}")


def set_ext_state(section: str, key: str, value: str, persist=False):
    """persist=True means the value should be stored and reloaded the next time REAPER is opened."""
    RPR_SetExtState(section, key, value, int(persist))


def has_ext_state(section: str, key: str) -> bool:
    return bool(RPR_HasExtState(section, key))


def delete_ext_state(section: str, key: str, persist=False):
    """persist=True means the value should remain deleted the next time REAPER is opened."""
    RPR_DeleteExtState(section, key, persist)


def get_cursor_position_sec() -> float:
    return RPR_GetCursorPosition()


def time_to_QN(t: float) -> float:
    return RPR_TimeMap2_timeToQN(0, t)

