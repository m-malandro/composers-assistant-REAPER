import os
########################################################################################################################
# REAPER USER SETTINGS:

# To decide what instrument a given Reaper track is, the program takes the first of these that matches any part of
# the track name.
# Example: If the track name is 'grand pianos', that corresponds to instrument 0, since 'piano' appears in the name.
# Example: If the track name is 'harpsichord ep1', that corresponds to instrument 4, since 'ep1' is the first matching
#          instrument in the name.
# You may edit this to reflect your preferred track naming conventions in Reaper.
INST_TO_MATCHING_STRINGS_RPR = {0: ['piano', 'key'],  # ac grand piano
                                1: ['bright'],  # bright ac piano
                                2: [],  # electric grand piano
                                3: ['honk'],  # honky-tonk piano
                                4: ['ep1'],  # electric piano 1
                                5: ['ep2'],  # electric piano 2
                                6: ['harpsi'],  # harpsichord
                                7: ['clav'],  # clavi
                                8: ['celest'],  # celesta
                                9: ['glock'],  # glock
                                10: ['box'],  # music box
                                11: ['vibra'],  # vibraphone
                                12: ['marimba'],  # marimba
                                13: ['xyl'],  # xylophone
                                14: ['bell', 'tubu'],  # tubular bells
                                15: ['dulcimer'],  # dulcimer
                                16: ['organ', 'dorg'],  # drawbar organ
                                17: ['porg'],  # percussive organ
                                18: ['rorg'],  # rock organ
                                19: ['corg'],  # church organ
                                20: ['reorg', 'reed'],  # reed organ
                                21: ['acc'],  # accordian
                                22: ['harmonica'],  # harmonica
                                23: ['tango'],  # tango accordian
                                24: ['nyl'],  # ac guitar (nylon)
                                25: ['steel g', 'sgtr', 'agtr'],  # ac guitar (steel)
                                26: ['jazz', 'jgtr'],  # elec guitar (jazz)
                                27: ['cgtr'],  # elec guitar (clean)
                                28: ['mute', 'mgtr'],  # elec guitar (muted)
                                29: ['ogtr', 'over'],  # overdriven guitar
                                30: ['gtr', 'guit', 'dist'],  # distortion guitar
                                31: ['harmon'],  # guitar harmonics ("harmon" doesn't conflict with "harmonica" above)
                                32: ['aco'],  # ac bass
                                33: ['finger'],  # electric bass (finger)
                                34: ['bass'],  # electric bass (pick)
                                35: ['fretless'],  # fretless bass
                                36: [],  # slap bass 1
                                37: ['slap'],  # slap bass 2
                                38: ['sbass1'],  # synth bass 1
                                39: ['sbass2'],  # synth bass 2
                                40: ['violin', 'vn1', 'vn2'],  # violin
                                41: ['viola', 'vn3'],  # viola
                                42: ['cell', 'vn4'],  # cello
                                43: ['contra', 'vn5'],  # contrabass
                                44: ['trem'],  # tremolo strings
                                45: ['pizz'],  # pizz strings
                                46: ['harp'],  # orchestral harp
                                47: ['timp'],  # timpani
                                48: ['str'],  # string ensemble 1
                                49: [],  # string ensemble 2
                                50: [],  # synth strings 1
                                51: [],  # synth strings 2
                                52: ['choir', 'aah'],  # choir aahs
                                53: ['ooh'],  # voice oohs
                                54: ['voice'],  # synth voice
                                55: ['hit', 'orch'],  # orchestra hit
                                56: ['trumpet', 'tp'],  # trumpet
                                57: ['trom'],  # trombone
                                58: ['tuba'],  # tuba
                                59: ['muted'],  # muted trumpet
                                60: ['french', 'horn', 'fh'],  # french horn
                                61: ['brass'],  # brass section
                                62: [],  # synth brass 1
                                63: [],  # synth brass 2
                                64: ['sax'],  # soprano sax
                                65: [],  # alto sax
                                66: [],  # tenor sax
                                67: [],  # baritone sax
                                68: ['oboe'],  # oboe
                                69: ['english horn', 'english'],  # english horn
                                70: ['bsn', 'baso'],  # bassoon
                                71: ['clarinet'],  # clarinet
                                72: ['picc'],  # piccolo
                                73: ['flute', 'fl1', 'fl2'],  # flute
                                74: ['recorder'],  # recorder
                                75: ['pan'],  # pan flute
                                76: ['bottle'],  # blown bottle
                                77: ['shak'],  # shakuhachi
                                78: ['whistle'],  # whistle
                                79: ['ocarina'],  # ocarina
                                80: ['square', 'lead1', 'ld1'],  # lead 1 (square)
                                81: ['saw', 'lead2', 'ld2'],  # lead 2 (sawtooth)
                                82: ['calli', 'lead3', 'ld3'],  # lead 3 (calliope)
                                83: ['chiff', 'lead4', 'ld4'],  # lead 4 (chiff)
                                84: ['chara', 'lead5', 'ld5'],  # lead 5 (charang)
                                85: ['lead 6', 'ld6'],  # lead 6 (voice)
                                86: ['fifth', 'lead7', 'ld7'],  # lead 7 (fifths)
                                87: ['lead8', 'ld8'],  # lead 8 (bass + lead)
                                88: ['pad1', 'new', 'age'],  # pad 1 (new age)
                                89: ['pad2', 'warm'],  # pad 2 (warm)
                                90: ['pad3', 'polys'],  # pad 3 (polysynth)
                                91: ['pad4', 'cpad'],  # pad 4 (choir)
                                92: ['pad5', 'bowed'],  # pad 5 (bowed)
                                93: ['pad6', 'metallic'],  # pad 6 (metallic)
                                94: ['pad7', 'halo'],  # pad 7 (halo)
                                95: ['pad8', 'sweep'],  # pad 8 (sweep)
                                96: ['fx1', 'rain'],  # FX 1 (rain)
                                97: ['fx2', 'soundtrack'],  # FX 2 (soundtrack)
                                98: ['fx3', 'crystal'],  # FX 3 (crystal)
                                99: ['fx4', 'atmos'],  # FX 4 (atmosphere)
                                100: ['fx5'],  # FX 5 (brightness)
                                101: ['fx6', 'goblin'],  # FX 6 (goblins)
                                102: ['fx7', 'echoes'],  # FX 7 (echoes)
                                103: ['fx8', 'sci'],  # FX 8 (sci-fi)
                                104: ['sitar'],  # sitar
                                105: ['banjo'],  # banjo
                                106: ['sham'],  # shamisen
                                107: ['koto'],  # koto
                                108: ['kali'],  # kalimba
                                109: ['bag', 'pipe'],  # bag pipe
                                110: ['fiddle'],  # fiddle
                                111: ['shan'],  # shanai
                                112: ['tink'],  # tinkle bell
                                113: ['agog'],  # agogo
                                114: ['steel'],  # steel drums
                                115: ['wood'],  # woodblock
                                116: ['taiko'],  # taiko
                                117: ['mtom', 'melodic tom'],  # melodic tom
                                118: ['synth drum', 'sdrum'],  # synth drum
                                119: ['rev'],  # reverse cymbal
                                120: ['fret'],  # guitar fret noise
                                121: ['breath'],  # breath noise
                                122: ['seashore'],  # seashore
                                123: ['bird', 'tweet'],  # bird tweet
                                124: ['phone'],  # telephone ring
                                125: ['heli'],  # helicopter
                                126: ['applause'],  # applause
                                127: ['gunshot'],  # gunshot
                                128: ['drum', 'kick', 'kik', 'sn', 'tom', 'hat', 'ride', 'crash', 'china', 'tambo']
                                # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                }

# Edit the following to force neural net outputs (via octave transposition) into the ranges defined below.
# Every note range must be at least one octave. Values represent closed intervals.
# The default values below should be good for most people, but you can change them if you want.
# (21, 108) is a standard 88-key grand piano.
# A typical 61 key keyboard would be (36, 96)
ACCEPTABLE_NOTE_RANGE_BY_INST_RPR = {0: (21, 108),  # ac grand piano
                                     1: (21, 108),  # bright ac piano
                                     2: (24, 96),  # electric grand piano
                                     3: (24, 96),  # honky-tonk piano
                                     4: (24, 96),  # electric piano 1
                                     5: (24, 96),  # electric piano 2
                                     6: (21, 89),  # harpsichord
                                     7: (21, 88),  # clavi
                                     8: (48, 96),  # celesta
                                     9: (67, 96),  # glock
                                     10: (48, 96),  # music box
                                     11: (53, 89),  # vibraphone
                                     12: (45, 96),  # marimba
                                     13: (53, 96),  # xylophone
                                     14: (43, 79),  # tubular bells
                                     15: (36, 96),  # dulcimer
                                     16: (36, 96),  # drawbar organ
                                     17: (36, 96),  # percussive organ
                                     18: (36, 96),  # rock organ
                                     19: (36, 96),  # church organ
                                     20: (36, 96),  # reed organ
                                     21: (36, 96),  # accordian
                                     22: (36, 96),  # harmonica
                                     23: (36, 96),  # tango accordian
                                     24: (36, 88),  # ac guitar (nylon)
                                     25: (36, 88),  # ac guitar (steel)
                                     26: (36, 88),  # elec guitar (jazz)
                                     27: (36, 88),  # elec guitar (clean)
                                     28: (36, 88),  # elec guitar (muted)
                                     29: (36, 88),  # overdriven guitar
                                     30: (36, 88),  # distortion guitar
                                     31: (36, 88),  # guitar harmonics
                                     32: (36, 72),  # ac bass
                                     33: (36, 72),  # electric bass (finger)
                                     34: (36, 72),  # electric bass (pick)
                                     35: (36, 72),  # fretless bass
                                     36: (36, 72),  # slap bass 1
                                     37: (36, 72),  # slap bass 2
                                     38: (36, 72),  # synth bass 1
                                     39: (36, 72),  # synth bass 2
                                     40: (55, 100),  # violin
                                     41: (48, 91),  # viola
                                     42: (36, 79),  # cello
                                     43: (36, 59),  # contrabass
                                     44: (36, 96),  # tremolo strings
                                     45: (36, 96),  # pizz strings
                                     46: (48, 100),  # orchestral harp
                                     47: (36, 81),  # timpani
                                     48: (36, 96),  # string ensemble 1
                                     49: (36, 96),  # string ensemble 2
                                     50: (36, 96),  # synth strings 1
                                     51: (36, 96),  # synth strings 2
                                     52: (36, 96),  # choir aahs
                                     53: (36, 96),  # voice oohs
                                     54: (36, 96),  # synth voice
                                     55: (36, 96),  # orchestra hit
                                     56: (52, 88),  # trumpet
                                     57: (36, 72),  # trombone
                                     58: (26, 58),  # tuba
                                     59: (52, 88),  # muted trumpet
                                     60: (36, 79),  # french horn
                                     61: (36, 88),  # brass section
                                     62: (36, 88),  # synth brass 1
                                     63: (36, 88),  # synth brass 2
                                     64: (36, 96),  # soprano sax
                                     65: (36, 96),  # alto sax
                                     66: (36, 96),  # tenor sax
                                     67: (36, 96),  # baritone sax
                                     68: (58, 91),  # oboe
                                     69: (52, 81),  # english horn
                                     70: (35, 74),  # bassoon
                                     71: (55, 96),  # clarinet
                                     72: (74, 108),  # piccolo
                                     73: (60, 96),  # flute
                                     74: (60, 96),  # recorder
                                     75: (48, 96),  # pan flute
                                     76: (36, 96),  # blown bottle
                                     77: (60, 96),  # shakuhachi
                                     78: (60, 96),  # whistle
                                     79: (60, 96),  # ocarina
                                     80: (36, 96),  # lead 1 (square)
                                     81: (36, 96),  # lead 2 (sawtooth)
                                     82: (36, 96),  # lead 3 (calliope)
                                     83: (36, 96),  # lead 4 (chiff)
                                     84: (36, 96),  # lead 5 (charang)
                                     85: (36, 96),  # lead 6 (voice)
                                     86: (36, 96),  # lead 7 (fifths)
                                     87: (36, 96),  # lead 8 (bass + lead)
                                     88: (36, 96),  # pad 1 (new age)
                                     89: (36, 96),  # pad 2 (warm)
                                     90: (36, 96),  # pad 3 (polysynth)
                                     91: (36, 96),  # pad 4 (choir)
                                     92: (36, 96),  # pad 5 (bowed)
                                     93: (36, 96),  # pad 6 (metallic)
                                     94: (36, 96),  # pad 7 (halo)
                                     95: (36, 96),  # pad 8 (sweep)
                                     96: (36, 96),  # FX 1 (rain)
                                     97: (36, 96),  # FX 2 (soundtrack)
                                     98: (36, 96),  # FX 3 (crystal)
                                     99: (36, 96),  # FX 4 (atmosphere)
                                     100: (36, 96),  # FX 5 (brightness)
                                     101: (36, 96),  # FX 6 (goblins)
                                     102: (36, 96),  # FX 7 (echoes)
                                     103: (36, 96),  # FX 8 (sci-fi)
                                     104: (36, 96),  # sitar
                                     105: (43, 96),  # banjo
                                     106: (36, 96),  # shamisen
                                     107: (36, 96),  # koto
                                     108: (36, 96),  # kalimba
                                     109: (36, 96),  # bag pipe
                                     110: (55, 100),  # fiddle
                                     111: (36, 96),  # shanai
                                     112: (36, 96),  # tinkle bell
                                     113: (36, 96),  # agogo
                                     114: (36, 96),  # steel drums
                                     115: (36, 96),  # woodblock
                                     116: (36, 96),  # taiko
                                     117: (36, 96),  # melodic tom
                                     118: (36, 96),  # synth drum
                                     119: (36, 96),  # reverse cymbal
                                     120: (36, 96),  # guitar fret noise
                                     121: (36, 96),  # breath noise
                                     122: (36, 96),  # seashore
                                     123: (36, 96),  # bird tweet
                                     124: (36, 96),  # telephone ring
                                     125: (36, 96),  # helicopter
                                     126: (36, 96),  # applause
                                     127: (36, 96),  # gunshot
                                     128: (0, 127)  # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                     }

########################################################################################################################
# NEURAL NET TRAINING SETTINGS
# Do not change any of the following unless you are finetuning a model or training a new model from scratch

# Change the following variable to the folder containing your training (or finetuning) MIDI.
PATH_TO_TRAIN_MIDI = r'C:\path\to\train\midi'
# Change the following variable to the folder that will hold the processed versions of the above MIDI.
# You can delete the files that get created in this folder after finishing your finetuning.
PATH_TO_PROCESSED_TRAIN_MIDI = r'C:\path\to\processed_train\midi'

# Even if you are not planning to run validation, you should put at least one midi file in PATH_TO_VAL_MIDI
PATH_TO_VAL_MIDI = r'C:\path\to\val\midi'
PATH_TO_PROCESSED_VAL_MIDI = r'C:\path\to\processed_val\midi'

# Even if you are not planning to run tests, you should put at least one midi file in PATH_TO_TEST_MIDI
PATH_TO_TEST_MIDI = r'C:\path\to\test\midi'
PATH_TO_PROCESSED_TEST_MIDI = r'C:\path\to\processed_test\midi'

# Empty folder needed for temporary storage during training and evaluation.
# If it doesn't exist, it will be created for you.
# You can delete every file in this folder after training and evaluation.
PATH_TO_TEMP_FILES = r'C:\path\to\temp\files'

FINETUNE_TASK = 'infill'  # Currently only 'infill' is available. The plan is to add additional tasks over time.

# Set UNJOINED = 1 to train/validate a basic event-based vocab.
# Set UNJOINED = 0 to train/validate a joined-event sentencepiece vocab learned from train data.
UNJOINED = 1

# set the following paths to empty folders if you are training from scratch;
# otherwise, make them the paths that you have already-pretrained or already-finetuned models in.
# Defaults for REAPER installation:
# UNJOINED_PATH_TO_MODELS = os.path.join('models', 'unjoined')
# JOINED_PATH_TO_MODELS = os.path.join('models', 'joined')
UNJOINED_PATH_TO_MODELS = os.path.join('models_permuted_labels', 'unjoined')
JOINED_PATH_TO_MODELS = os.path.join('models_permuted_labels', 'joined')

# During training, validation, and testing, as a form of data augmentation, we transpose our songs
# randomly by an integer amount in the closed interval [AUG_TRANS_MIN, AUG_TRANS_MAX]
# Defaults:
# AUG_TRANS_MIN, AUG_TRANS_MAX = -5, 6
AUG_TRANS_MIN = -5
AUG_TRANS_MAX = 6

# MAX_LEN defines the maximum number of tokens to train the net on per example; You should set MAX_LEN >= 512.
# If you want to set MAX_LEN to something higher than 4096, then you will need to edit pretrain_model.py
# and train a new model from scratch.
# NOTE: MAX_LEN = 1024 enables fine-tuning of a 384-dim 10-layer model on a 6 GB 1060 GTX.
# NOTE: MAX_LEN = 1650 enables fine-tuning of a 384-dim 10-layer model on a 12 GB card.
# with finetune_model.BATCH_SIZE = (1, X). If you have more memory available, you can set MAX_LEN higher.
# Applies to FINETUNE_TASK = 'infill'
MAX_LEN = 1650

# TODO maybe - write code for training for 'arrange' functionality
# Applies to FINETUNE_TASK = 'arrange'
# MAX_INPUT_LEN_ARRANGE = 1350
# MAX_OUTPUT_LEN_ARRANGE = 1900

########################################################################################################################
# MORE NEURAL NET TRAINING SETTINGS
# Do not change any of the following unless you are training a new model from scratch
QUANTIZE = (8, 6)

# Our primary joined-vocabulary model is SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS.
# Other SentencePiece vocabulary models are currently incompatible with our REAPER scripts.
# Other SentencePiece vocabulary models are also incompatible with v2 and beyond. They are just relics of
# past experiments.
SPM_TRAIN_MODEL_WITH_NOTE_OFFS = False
SPM_TRAIN_MODEL_WITH_NOTE_LENGTHS = False
SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS = True
SPM_TRAIN_MODEL_WITHOUT_NOTE_OFFS = False

# Train each sentencepiece model from approximately this number of examples.
# Each example is a string representing a single instrument in a single measure.
# Default: 5000000; The default amount requires a significant amount of RAM + CPU.
SPM_NUM_EXAMPLES = 5000000

# Number of chords/short musical phrases for sentencepiece vocab to learn
# Actual vocab size will be slightly higher (usually about 1500 higher; the exact amount depends on QUANTIZE)
N_PHRASES_FOR_VOCAB = 15000

# neural net parameters for training a new neural net from scratch;
# additional parameters may be edited in pretrain_model.py.
# D_MODEL = 384
# N_LAYERS = 10
# N_HEADS = 8

D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8

# number of epochs to pretrain on inputs of length 512
N_EPOCHS_SHORT = 6

# SPM_TYPE: Choose one of the following to use to train your model.
# 'duration' (available if SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS was True; default)
# 'length' (available if SPM_TRAIN_MODEL_WITH_NOTE_LENGTHS was True; not recommended; use 'duration' instead)
# 'include_note_offs' (available if SPM_TRAIN_MODEL_WITH_NOTE_OFFS was True; not recommended; use 'duration' instead)
# 'exclude_note_offs' (available if SPM_TRAIN_MODEL_WITHOUT_NOTE_OFFS was True)
# 'unjoined_include_note_duration_commands' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_include_note_length' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_include_note_offs' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_exclude_note_offs' (always available; basic sentencepiece-like model containing no event merges)
if UNJOINED:
    SPM_TYPE = 'unjoined_include_note_duration_commands'  # default: 'unjoined_include_note_duration_commands'
else:
    SPM_TYPE = 'duration'  # default: 'duration'

# Map drum notes to -1 to disable them. Set this map up before running preprocess_midi.py.
SIMPLIFIED_DRUM_MAP = {0: -1,
                       1: -1,
                       2: -1,
                       3: -1,
                       4: -1,
                       5: -1,
                       6: -1,
                       7: -1,
                       8: -1,
                       9: -1,
                       10: -1,
                       11: -1,
                       12: -1,
                       13: -1,
                       14: -1,
                       15: -1,
                       16: -1,
                       17: -1,
                       18: -1,
                       19: -1,
                       20: -1,
                       21: -1,
                       22: -1,
                       23: -1,
                       24: -1,
                       25: -1,
                       26: 38,  # snap
                       27: 36,  # boopy kick
                       28: 38,  # cardboard
                       29: 36,  # record scratch
                       30: 36,  # record scratch 2
                       31: 46,  # stick click
                       32: 50,  # L bongo?
                       33: 37,  # rim click?
                       34: 53,  # telephone ding
                       35: 36,  # kick
                       36: 36,  # kick - primary
                       37: 37,  # sidestick
                       38: 38,  # snare - primary
                       39: 39,  # clap
                       40: 38,  # snare
                       41: 41,  # lowest tom
                       42: 42,  # closed HH - primary
                       43: 43,  # tom
                       44: 44,  # pedal HH
                       45: 45,  # tom
                       46: 46,  # open HH - primary
                       47: 47,  # tom
                       48: 48,  # tom
                       49: 49,  # crash - primary
                       50: 50,  # highest tom
                       51: 51,  # ride
                       52: 52,  # china
                       53: 53,  # ride bell
                       54: 54,  # tambourine
                       55: 55,  # crash (splash sometimes)
                       56: 56,  # cowbell
                       57: 57,  # crash 2
                       58: 73,  # frogs? (vibraslap)
                       59: 51,  # ride 2
                       60: 50,  # hi bongo
                       61: 47,  # low bongo
                       62: 48,  # mute hi conga
                       63: 45,  # open hi conga
                       64: 43,  # low conga
                       65: 48,  # hi timbale
                       66: 47,  # low timbale
                       67: 53,  # hi agogo (bell)
                       68: 51,  # low agogo (bell)
                       69: 69,  # shaker
                       70: 70,  # maraca
                       71: -1,  # whistle
                       72: -1,  # whistle
                       73: 73,  # short guiro (frog)
                       74: 73,  # long guiro (frog)
                       75: 76,  # claves (woodblockish)
                       76: 76,  # hi wood block
                       77: 76,  # low wood block
                       78: -1,  # mute cuica (honky voice?)
                       79: -1,  # open cuica (honky voice?)
                       80: 80,  # muted triangle
                       81: 81,  # unmuted ("open") triangle
                       82: 69,  # shaker
                       83: 46,  # sleigh bells
                       84: -1,  # small chimes
                       85: 39,  # snap
                       86: 48,  # hi tom
                       87: 47,  # lower tom
                       88: -1,
                       89: -1,
                       90: -1,
                       91: -1,
                       92: -1,
                       93: -1,
                       94: -1,
                       95: -1,
                       96: -1,
                       97: -1,
                       98: -1,
                       99: -1,
                       100: -1,
                       101: -1,
                       102: -1,
                       103: -1,
                       104: -1,
                       105: -1,
                       106: -1,
                       107: -1,
                       108: -1,
                       109: -1,
                       110: -1,
                       111: -1,
                       112: -1,
                       113: -1,
                       114: -1,
                       115: -1,
                       116: -1,
                       117: -1,
                       118: -1,
                       119: -1,
                       120: -1,
                       121: -1,
                       122: -1,
                       123: -1,
                       124: -1,
                       125: -1,
                       126: -1,
                       127: -1,
                       }

# every note range must be at least one octave
# values represent closed intervals
# (21, 108) is a standard 88-key grand piano
# Set this up before running spm_create_train_data.py.
ACCEPTABLE_NOTE_RANGE_BY_INST_TRAIN_TEST = {0: (0, 127),  # ac grand piano
                                            1: (0, 127),  # bright ac piano
                                            2: (0, 127),  # electric grand piano
                                            3: (0, 127),  # honky-tonk piano
                                            4: (0, 127),  # electric piano 1
                                            5: (0, 127),  # electric piano 2
                                            6: (0, 127),  # harpsichord
                                            7: (0, 127),  # clavi
                                            8: (0, 127),  # celesta
                                            9: (0, 127),  # glock
                                            10: (0, 127),  # music box
                                            11: (0, 127),  # vibraphone
                                            12: (0, 127),  # marimba
                                            13: (0, 127),  # xylophone
                                            14: (0, 127),  # tubular bells
                                            15: (0, 127),  # dulcimer
                                            16: (0, 127),  # drawbar organ
                                            17: (0, 127),  # percussive organ
                                            18: (0, 127),  # rock organ
                                            19: (0, 127),  # church organ
                                            20: (0, 127),  # reed organ
                                            21: (0, 127),  # accordian
                                            22: (0, 127),  # harmonica
                                            23: (0, 127),  # tango accordian
                                            24: (0, 127),  # ac guitar (nylon)
                                            25: (0, 127),  # ac guitar (steel)
                                            26: (0, 127),  # elec guitar (jazz)
                                            27: (0, 127),  # elec guitar (clean)
                                            28: (0, 127),  # elec guitar (muted)
                                            29: (0, 127),  # overdriven guitar
                                            30: (0, 127),  # distortion guitar
                                            31: (0, 127),  # guitar harmonics
                                            32: (0, 127),  # ac bass
                                            33: (0, 127),  # electric bass (finger)
                                            34: (0, 127),  # electric bass (pick)
                                            35: (0, 127),  # fretless bass
                                            36: (0, 127),  # slap bass 1
                                            37: (0, 127),  # slap bass 2
                                            38: (0, 127),  # synth bass 1
                                            39: (0, 127),  # synth bass 2
                                            40: (0, 127),  # violin
                                            41: (0, 127),  # viola
                                            42: (0, 127),  # cello
                                            43: (0, 127),  # contrabass
                                            44: (0, 127),  # tremolo strings
                                            45: (0, 127),  # pizz strings
                                            46: (0, 127),  # orchestral harp
                                            47: (0, 127),  # timpani
                                            48: (0, 127),  # string ensemble 1
                                            49: (0, 127),  # string ensemble 2
                                            50: (0, 127),  # synth strings 1
                                            51: (0, 127),  # synth strings 2
                                            52: (0, 127),  # choir aahs
                                            53: (0, 127),  # voice oohs
                                            54: (0, 127),  # synth voice
                                            55: (0, 127),  # orchestra hit
                                            56: (0, 127),  # trumpet
                                            57: (0, 127),  # trombone
                                            58: (0, 127),  # tuba
                                            59: (0, 127),  # muted trumpet
                                            60: (0, 127),  # french horn
                                            61: (0, 127),  # brass section
                                            62: (0, 127),  # synth brass 1
                                            63: (0, 127),  # synth brass 2
                                            64: (0, 127),  # soprano sax
                                            65: (0, 127),  # alto sax
                                            66: (0, 127),  # tenor sax
                                            67: (0, 127),  # baritone sax
                                            68: (0, 127),  # oboe
                                            69: (0, 127),  # english horn
                                            70: (0, 127),  # bassoon
                                            71: (0, 127),  # clarinet
                                            72: (0, 127),  # piccolo
                                            73: (0, 127),  # flute
                                            74: (0, 127),  # recorder
                                            75: (0, 127),  # pan flute
                                            76: (0, 127),  # blown bottle
                                            77: (0, 127),  # shakuhachi
                                            78: (0, 127),  # whistle
                                            79: (0, 127),  # ocarina
                                            80: (0, 127),  # lead 1 (square)
                                            81: (0, 127),  # lead 2 (sawtooth)
                                            82: (0, 127),  # lead 3 (calliope)
                                            83: (0, 127),  # lead 4 (chiff)
                                            84: (0, 127),  # lead 5 (charang)
                                            85: (0, 127),  # lead 6 (voice)
                                            86: (0, 127),  # lead 7 (fifths)
                                            87: (0, 127),  # lead 8 (bass + lead)
                                            88: (0, 127),  # pad 1 (new age)
                                            89: (0, 127),  # pad 2 (warm)
                                            90: (0, 127),  # pad 3 (polysynth)
                                            91: (0, 127),  # pad 4 (choir)
                                            92: (0, 127),  # pad 5 (bowed)
                                            93: (0, 127),  # pad 6 (metallic)
                                            94: (0, 127),  # pad 7 (halo)
                                            95: (0, 127),  # pad 8 (sweep)
                                            96: (0, 127),  # FX 1 (rain)
                                            97: (0, 127),  # FX 2 (soundtrack)
                                            98: (0, 127),  # FX 3 (crystal)
                                            99: (0, 127),  # FX 4 (atmosphere)
                                            100: (0, 127),  # FX 5 (brightness)
                                            101: (0, 127),  # FX 6 (goblins)
                                            102: (0, 127),  # FX 7 (echoes)
                                            103: (0, 127),  # FX 8 (sci-fi)
                                            104: (0, 127),  # sitar
                                            105: (0, 127),  # banjo
                                            106: (0, 127),  # shamisen
                                            107: (0, 127),  # koto
                                            108: (0, 127),  # kalimba
                                            109: (0, 127),  # bag pipe
                                            110: (0, 127),  # fiddle
                                            111: (0, 127),  # shanai
                                            112: (0, 127),  # tinkle bell
                                            113: (0, 127),  # agogo
                                            114: (0, 127),  # steel drums
                                            115: (0, 127),  # woodblock
                                            116: (0, 127),  # taiko
                                            117: (0, 127),  # melodic tom
                                            118: (0, 127),  # synth drum
                                            119: (0, 127),  # reverse cymbal
                                            120: (0, 127),  # guitar fret noise
                                            121: (0, 127),  # breath noise
                                            122: (0, 127),  # seashore
                                            123: (0, 127),  # bird tweet
                                            124: (0, 127),  # telephone ring
                                            125: (0, 127),  # helicopter
                                            126: (0, 127),  # applause
                                            127: (0, 127),  # gunshot
                                            128: (0, 127)
                                            # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                            }

# HORIZ_NOTE_ONSET_DENSITY_SLICES defines a set of intervals via bisect.bisect.
# EXAMPLE: HORIZ_NOTE_ONSET_DENSITY_SLICES = [1, 2, 4] means the four intervals would be
#          (-inf, 1), [1, 2), [2, 4), [4, inf)
# Actual values used for the intervals in the default below, by interval index:
#  0: less frequent than half notes (on avg) (so usually like whole notes,
#                                             or sometimes a random 1 note of an arbitrary length)
#  1: half notes to not quite quarter notes
#  2: quarter notes to not quite eighth notes
#  3: eighth notes to not quite 16th notes
#  4: 16th notes to a little more than 16th notes
#  5: more than more than 16th notes
HORIZ_NOTE_ONSET_DENSITY_SLICES = [0.5, 1, 2, 4, 4.5]  # default: [0.5, 1, 2, 4, 4.5]

# VERT_NOTE_ONSET_DENSITY_SLICES defines a set of intervals via bisect.bisect.
# Actual values for the intervals used in the default below, by interval index, in mathematical interval notation:
#  0: mono
#  1: (mono, power chords]
#  2: (power chords, triads] on avg
#  3: (triads, 4-note chords] on avg
#  4: (4-note chords, inf) on avg
VERT_NOTE_ONSET_DENSITY_SLICES = [1, 2, 3, 4]  # default: [1, 2, 3, 4]
VERT_NOTE_ONSET_N_PITCH_CLASSES_SLICES = [1, 2, 3, 4]  # default: [1, 2, 3, 4]

# Similar to HORIZ_NOTE_ONSET_DENSITY_SLICES and VERT_NOTE_ONSET_DENSITY_SLICES
PITCH_HIST_STEP_SLICES = [0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
PITCH_HIST_LEAP_SLICES = [0.01, 0.2, 0.4, 0.6, 0.8, 0.99]

HORIZ_NOTE_ONSET_IRREGULARITY_SLICES = [0.01, 0.14, 0.4]  # this idea could use some refinement in future work

# HORIZ_NOTE_ONSET_DENSITY_STD_SLICES = [0.01, 0.3, 0.5]  # not used
HORIZ_NOTE_ONSET_DENSITY_DIVERSITY_PERCENTAGE_SLICES = [0.01, 0.25, 0.5]  # this idea is undercooked at the moment
# VERT_NOTE_ONSET_DENSITY_STD_SLICES = []  # not used
# PITCH_HIST_STEP_STD_SLICES = []  # not used
# PITCH_HIST_LEAP_STD_SLICES = []  # not used

# don't change this
if UNJOINED:
    PATH_TO_MODELS = UNJOINED_PATH_TO_MODELS
else:
    PATH_TO_MODELS = JOINED_PATH_TO_MODELS


# TODO: edit/delete before release
# overrides for the code author. You can ignore these.
# PATH_TO_TRAIN_MIDI = r''
# PATH_TO_PROCESSED_TRAIN_MIDI = r''
# PATH_TO_VAL_MIDI = ''
# PATH_TO_TEST_MIDI = ''
# PATH_TO_PROCESSED_VAL_MIDI = r'C:\datasets\lakh\b_lakh_midi_val_processed'
# PATH_TO_PROCESSED_TEST_MIDI = r'C:\datasets\lakh\c_lakh_midi_test_processed'
# PATH_TO_TEMP_FILES = r''
# PATH_TO_TEMP_FILES = r''
# PATH_TO_MODELS = r''
