import spm_train_functions
import midisong as ms
import constants


class UnjoinedTokenizer(object):
    def __init__(self, mode):
        if mode not in ('unjoined_include_note_length',
                        'unjoined_include_note_duration_commands',
                        'unjoined_include_note_offs',
                        'unjoined_exclude_note_offs'):
            raise ValueError('mode {} not recognized'.format(mode))

        self.mode = mode

        user_defined_symbols = spm_train_functions.get_user_defined_symbols()
        no_joins_vocab = user_defined_symbols

        no_joins_vocab.insert(0, ';<pad>')
        no_joins_vocab.insert(0, ';<eos>')
        no_joins_vocab.insert(0, ';<bos>')
        no_joins_vocab.insert(0, ';<unk>')

        if 'include_note_offs' in self.mode:
            for x in range(128):
                no_joins_vocab.append(';T:{}'.format(x))  # tie symbols

        for x in range(128):
            no_joins_vocab.append(';N:{}'.format(x))  # note on symbols

        if 'include_note_length' in self.mode:
            max_note_length = 8 * ms.extended_lcm(constants.QUANTIZE)  # 8 QN's max length
            for x in range(0, max_note_length + 1):
                no_joins_vocab.append(':{}'.format(x))

        if 'include_note_duration_commands' in self.mode:
            max_note_length = 8 * ms.extended_lcm(constants.QUANTIZE)  # 8 QN's max length
            for x in range(0, max_note_length + 1):
                no_joins_vocab.append(';d:{}'.format(x))

        for x in range(128):
            no_joins_vocab.append(';D:{}'.format(x))  # drum hit symbols

        if 'include_note_offs' in self.mode:
            for x in range(128):
                no_joins_vocab.append(';/N:{}'.format(x))  # note off symbols

        for x in range(1, 8 * ms.extended_lcm(constants.QUANTIZE) + 1):
            no_joins_vocab.append(';w:{}'.format(x))  # wait symbols

        self.vocab_to_int = {}
        self.int_to_vocab = {}
        for i, x in enumerate(no_joins_vocab):
            self.int_to_vocab[i] = x
            self.vocab_to_int[x] = i

    def pad_id(self):
        return self.vocab_to_int[';<pad>']

    def eos_id(self):
        return self.vocab_to_int[';<eos>']

    def bos_id(self):
        return self.vocab_to_int[';<bos>']

    def unk_id(self):
        return self.vocab_to_int[';<unk>']

    # test written
    def Encode(self, s: "str or list or tuple"):
        if type(s) != str:
            return self._EncodeList(s)
        instructions = s.split(';')
        instructions = [';'+i for i in instructions if i]
        if 'length' not in self.mode:
            return [self.vocab_to_int.get(i, self.unk_id()) for i in instructions]
        else:
            res = []
            for e in instructions:
                if e[:2] == ';N':
                    parts = e.split(':')
                    note_cmd = parts[0] + ':' + parts[1]
                    length = ':' + parts[2]
                    res.append(self.vocab_to_int[note_cmd])
                    res.append(self.vocab_to_int[length])
                else:
                    res.append(self.vocab_to_int.get(e, self.unk_id()))
            return res

    def _EncodeList(self, L):
        res = [self.Encode(x) for x in L]
        return res

    def encode(self, s):
        return self.Encode(s)

    def Decode(self, L: list or int):
        if type(L) == int:
            return self.int_to_vocab[L]
        else:
            return ''.join([self.int_to_vocab[x] for x in L])

    def decode(self, L):
        return self.Decode(L)

    def vocab_size(self):
        return len(self.vocab_to_int)

# vocab description for paper:
# <unk>
# <bos>
# <eos>
# <pad>
#
# ;B:0
# thru
# ;B:7
#
# ;M:0
# thru
# ;M:7
#
# ;L:1
# thru
# ;L:192
#
# ;I:0
# thru
# ;I:257
#
# ;R:1
# thru
# ;R:63
#
# ;<extra_id_0>
# thru
# ;<extra_id_255>
#
# ;<mono>
# ;<poly>
#
# ;<instruction_0>
# thru
# ;<instruction_511>
#
# ;N:0
# thru
# ;N:127
#
# ;d:0
# thru
# ;d:192
#
# ;D:0
# thru
# ;D:127
#
# ;w:1
# thru
# ;w:192
