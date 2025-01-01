import os.path
import constants as cs


def spm_type_to_note_off_treatment(s):
    if 'length' in s:
        return 'length'
    elif 'include_note_offs' in s:
        return 'include'
    elif 'exclude_note_offs' in s:
        return 'exclude'
    elif 'duration' in s:
        return 'duration'
    else:
        raise ValueError('unrecognized SPM type: {}'.format(s))


def get_tokenizer():
    if cs.SPM_TYPE == 'length':
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm', 'spm_incl_note_lengths.model'))
    elif cs.SPM_TYPE == 'duration':
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm', 'spm_incl_note_duration_commands.model'))
    elif cs.SPM_TYPE == 'include_note_offs':
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm', 'spm_incl_note_offs.model'))
    elif cs.SPM_TYPE == 'exclude_note_offs':
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(os.path.join(cs.JOINED_PATH_TO_MODELS, 'spm', 'spm_excl_note_offs.model'))
    elif cs.SPM_TYPE in ('unjoined_include_note_length',
                         'unjoined_include_note_duration_commands',
                         'unjoined_include_note_offs',
                         'unjoined_exclude_note_offs'):
        import unjoined_vocab_tokenizer
        tokenizer = unjoined_vocab_tokenizer.UnjoinedTokenizer(mode=cs.SPM_TYPE)
    else:
        raise ValueError('constant SPM_TYPE unrecognized:{}'.format(cs.SPM_TYPE))
    return tokenizer
