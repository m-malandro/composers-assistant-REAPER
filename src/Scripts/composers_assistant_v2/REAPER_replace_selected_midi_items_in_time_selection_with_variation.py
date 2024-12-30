import rpr_ca_functions as fn
from reaper_python import *
import sys


def patch_stdout_stderr_open():
    global open, original_open, reaper_console

    class ReaperConsole:
        def write(self, output):
            RPR_ShowConsoleMsg(output)

        def flush(self):
            pass

        def close(self):
            pass

    reaper_console = ReaperConsole()

    sys.stdout = reaper_console
    sys.stderr = reaper_console

    original_open = open
    open = lambda *args, **kwargs: reaper_console


patch_stdout_stderr_open()

def go():
    options = fn.get_global_options()
    if fn.DEBUG or options.display_track_to_MIDI_inst:
        RPR_ClearConsole()

    fn.create_and_write_variation_for_time_selection(n_cycles=3, mask_pattern_type=options.variation_alg,
                                                     create_variation_only_for_selected_midi_items=True,
                                                     new_notes_are_selected=options.generated_notes_are_selected,
                                                     temperature=options.temperature,
                                                     do_rhythmic_conditioning=options.do_rhythm_conditioning,
                                                     rhythmic_conditioning_type=options.rhythm_conditioning_type,
                                                     do_note_range_conditioning_by_measure=options.do_note_range_conditioning,
                                                     note_range_conditioning_type=options.note_range_conditioning_type,
                                                     display_track_to_MIDI_inst=options.display_track_to_MIDI_inst,
                                                     display_warnings=options.display_warnings)


if __name__ == '__main__':
    go()

RPR_Undo_OnStateChange('Variation')
