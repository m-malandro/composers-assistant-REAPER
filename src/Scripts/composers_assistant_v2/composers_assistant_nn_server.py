import os
import constants as cs
import unjoined_vocab_tokenizer as ujt
import preprocessing_functions as pre
import re
print('Composer Assistant v2 neural net server starting...')

try:
    import torch
except Exception as E:
    print(E)
    input('"torch" module not installed. Close this window, install pytorch, and try again. You may press the enter key to close this window.')
    assert False

try:
    import transformers
except Exception as E:
    print(E)
    input('"transformers" module not installed. Close this window, install transformers, and try again. You may press the enter key to close this window.')
    assert False

try:
    import sentencepiece as spm
    spm_installed = True
except Exception as E:
    spm_installed = False

# change the following function if model changes
SPM_PATH = os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm', 'spm_incl_note_duration_commands.model')
if spm_installed and os.path.exists(SPM_PATH):
    SPM_TOKENIZER = spm.SentencePieceProcessor(SPM_PATH)
else:
    SPM_TOKENIZER = None
UNJOINED_TOKENIZER = ujt.UnjoinedTokenizer('unjoined_include_note_duration_commands')

MAX_NN_LENGTH = cs.MAX_LEN
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

LAST_CALL = ''
LAST_OUTPUTS = set()

DEBUG = False


def normalize_requests(input_s):

    def norm_extra_id(s):
        first_loc = s.find('<extra_id_')
        if first_loc != -1:
            second_loc = s.find('>', first_loc)
            s = s[:first_loc] + '<e>' + s[second_loc + 1:]
            return norm_extra_id(s)
        else:
            return s

    def norm_measure(s):
        first_loc = s.find(';M:')
        if first_loc != -1:
            second_loc = s.find(';', first_loc + 1)
            if second_loc == -1:
                s = s[:first_loc] + '<M>'
            else:
                s = s[:first_loc] + '<M>' + s[second_loc:]
            return norm_measure(s)
        else:
            return s

    return norm_measure(norm_extra_id(input_s))


def load_models(task):
    # joined model
    model_path = os.path.join(cs.JOINED_PATH_TO_MODELS, str(task), 'finetuned_epoch_XXX_0', 'model')
    if os.path.exists(model_path) and SPM_TOKENIZER is not None:
        print('Loading neural net from: {}'.format(model_path))
        M = transformers.T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        M.eval()
        # print(M.num_parameters(), 'parameters')
    else:
        M = None

    # unjoined model
    model_path = os.path.join(cs.UNJOINED_PATH_TO_MODELS, str(task), 'finetuned_epoch_49_0', 'model')
    # model_path = os.path.join('_internal', 'models_permuted_labels', 'unjoined', 'infill', 'finetuned_epoch_49_0', 'model')  # for pyinstaller exe version
    
    if os.path.exists(model_path):
        print('Loading neural net from: {}'.format(model_path))
        M2 = transformers.T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        M2.eval()
        # print(M2.num_parameters(), 'parameters')
    else:
        M2 = None

    if M is not None or M2 is not None:
        return M, M2

    raise Exception('No finetuned models found! Aborting.')


def get_n_measures(s: str):
    return s.count(';M')


def choose_model_and_tokenizer_infill(s: str, has_fully_masked_inst: bool):
    if MODELS[f'unjoined infill'] is None:
        return MODELS[f'spm infill'], SPM_TOKENIZER, 'spm'
    if MODELS[f'spm infill'] is None:
        return MODELS[f'unjoined infill'], UNJOINED_TOKENIZER, 'unjoined'

    if len(UNJOINED_TOKENIZER.encode(s)) < 1024 and (get_n_measures(s) <= 9 or has_fully_masked_inst):
        M = MODELS['unjoined infill']
        tokenizer = UNJOINED_TOKENIZER
        str_description = 'unjoined'
        if DEBUG:
            print('using unjoined tokenizer and model')
    else:
        M = MODELS['spm infill']
        tokenizer = SPM_TOKENIZER
        str_description = 'spm'
        if DEBUG:
            print('using SPM tokenizer and model')

    return M, tokenizer, str_description


def call_nn_infill(s, S, use_sampling=True, min_length=10, enc_no_repeat_ngram_size=0,
                   has_fully_masked_inst=False, temperature=1.0) -> str:
    """s a string input to a nn, not yet tokenized"""
    global LAST_CALL, LAST_OUTPUTS

    s_request_normalized = normalize_requests(s)

    # print(S)
    S = pre.midisongbymeasure_from_save_dict(S)

    # print('request normalized:', s_request_normalized)
    # print('is the same as previous', s_request_normalized == LAST_CALL)

    if s_request_normalized != LAST_CALL:
        LAST_OUTPUTS = set()

    # choose model to use
    M, tokenizer, str_tok = choose_model_and_tokenizer_infill(s=s, has_fully_masked_inst=has_fully_masked_inst)

    # stretch enc_no_repeat_ngram_size if using unjoined tokenizer
    # if str_tok == 'unjoined' and enc_no_repeat_ngram_size:
    #     if enc_no_repeat_ngram_size == 3:
    #         enc_no_repeat_ngram_size = 3
    #     elif enc_no_repeat_ngram_size == 4:
    #         enc_no_repeat_ngram_size = 5
    if DEBUG:
        print(f'no_repeat_ngram_size = {enc_no_repeat_ngram_size}, temperature={temperature}')

    if use_sampling == 'None' or use_sampling is None:
        # then use greedy decoding for the first attempt
        use_sampling = len(LAST_OUTPUTS) != 0
        if DEBUG:
            if use_sampling:
                print('using top-p sampling')
            else:
                print('using greedy decoding')

    # if DEBUG:
    #     print('input: ', s)
    L = tokenizer.Encode(s)

    print(f'NN input (len {len(L)})')

    if len(L) > MAX_NN_LENGTH:
        print('WARNING: neural net input is too long. If you are unhappy with the output, '
              'try again with fewer measures selected.')

    input_ids = torch.stack([torch.tensor(L, dtype=torch.long, device=DEVICE)])

    done = False
    len_mult = 1

    if len(LAST_OUTPUTS) > 0:
        print("Attempting to avoid previous {} outputs".format(len(LAST_OUTPUTS)))

    # forced_ids = TOKENIZER.Encode(nns.extract_extra_ids(s))
    # print('forcing: ', TOKENIZER.Decode(forced_ids))

    temperature_multiplier = [1.0, 1.05, 1.10, 1.15, 1.25, 1.5, 1.75, 2.0, 2.5]
    attempt_index = 0

    while not done:
        top_p = .05/49*(len_mult - 1) + .95  # goes from (1, .95) to (50, 1.0)
        top_p = 0.85  # override
        gend = M.generate(input_ids=input_ids,
                          num_return_sequences=1,
                          do_sample=use_sampling,
                          temperature=temperature * temperature_multiplier[attempt_index],
                          # remove_invalid_values=True,
                          # top_k=100,
                          top_p=top_p,
                          min_length=min_length,
                          max_new_tokens=2*1650*len_mult,
                          decoder_start_token_id=tokenizer.pad_id(),
                          pad_token_id=tokenizer.pad_id(),
                          bos_token_id=tokenizer.bos_id(),
                          eos_token_id=tokenizer.eos_id(),
                          use_cache=True,
                          # force_words_ids=forced_ids,
                          encoder_no_repeat_ngram_size=enc_no_repeat_ngram_size,
                          # repetition_penalty=1.01
                          )

        this_candidate = gend[0][1:]
        this_candidate_tokenized = [x.item() for x in this_candidate]
        this_candidate = tokenizer.Decode(this_candidate_tokenized)

        this_candidate_normalized = normalize_requests(this_candidate)
        if this_candidate_normalized not in LAST_OUTPUTS:
            done = True

        if len_mult >= 9:
            done = True

        if not done:
            len_mult += 1
            attempt_index += 1
            print('Trying again: Attempt', len_mult, '(max 9)')

    LAST_CALL = s_request_normalized
    LAST_OUTPUTS.add(this_candidate_normalized)

    if DEBUG:
        print(f'NN output (len {len(this_candidate_tokenized)})')

    return this_candidate


if __name__ == '__main__':
    MODELS = {}
    MODELS['spm infill'], MODELS['unjoined infill'] = load_models(task='infill')

    from xmlrpc.server import SimpleXMLRPCServer
    SERVER = SimpleXMLRPCServer(('127.0.0.1', 3456), logRequests=True)
    SERVER.register_function(call_nn_infill)
    if str(DEVICE) == 'cuda':
        str_device = 'GPU'
    else:
        str_device = 'CPU'
    print('NN server running on device: {}. Press ctrl+c or close this window to shut it down.'.format(str_device))

    try:
        SERVER.serve_forever(poll_interval=0.01)
    except KeyboardInterrupt:
        print('NN server shutting down...')
