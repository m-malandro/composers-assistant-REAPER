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

    nn_input = fn.get_nn_input_from_project(mask_empty_midi_items=True, mask_selected_midi_items=True,
                                            do_rhythmic_conditioning=options.do_rhythm_conditioning,
                                            rhythmic_conditioning_type=options.rhythm_conditioning_type,
                                            do_note_range_conditioning_by_measure=options.do_note_range_conditioning,
                                            note_range_conditioning_type=options.note_range_conditioning_type,
                                            display_track_to_MIDI_inst=options.display_track_to_MIDI_inst,
                                            display_warnings=options.display_warnings)
    if nn_input.continue_:
        if fn.DEBUG:
            print('calling NN with input:')
            print(nn_input.nn_input_string)
        use_sampling = "None" if not fn.ALWAYS_TOP_P else True
        nn_output = fn.call_nn_infill(s=nn_input.nn_input_string,
                                      S=nn_input.S,
                                      use_sampling=use_sampling,
                                      min_length=2,
                                      enc_no_repeat_ngram_size=0,
                                      has_fully_masked_inst=nn_input.has_fully_masked_inst,
                                      temperature=options.temperature)
        if fn.DEBUG:
            print('got nn output: ', nn_output)
        fn.write_nn_output_to_project(nn_output=nn_output, nn_input_obj=nn_input,
                                      notes_are_selected=options.generated_notes_are_selected,
                                      use_vels_from_tr_measures=options.do_rhythm_conditioning,
                                      )


if __name__ == '__main__':
    go()

RPR_Undo_OnStateChange('Infill')
