import collections
import encoding_functions as enc


def parse_single_instruction(s: str, allow_NXY=False) -> str:
    """s a string, output from a neural net, intended to be a midi-like instruction.
    This function returns one of:
    'M:X',
    'L:X',
    'B:X',
    'I:X',
    'R:X',
    'w:X',
    'd:X',
    'p:X',
    'N:X',
    'N:X:Y' (if allowed),
    '/N:X',
    'D:X',
    any instruction starting with '<' (unchanged),
    '' (for none of the above)
    """
    # deal with empty inputs and special strings first
    if not s:
        return ''
    elif s[0] == '<':
        return s

    if s[0] in ('M', 'L', 'B', 'I', 'R', 'w', 'd', 'p', 'D'):
        res = s[0] + ':' + s.split(':')[1]
    elif s[0] == 'N':
        if not allow_NXY:
            res = s[0] + ':' + s.split(':')[1]
        else:
            s_split = s.split(':')
            res = s[0] + ':' + s_split[1]
            if len(s_split) > 2:
                res += ':' + s_split[2]
    elif s[:2] == '/N':
        res = '/N' + ':' + s.split(':')[1]
    else:
        res = ''

    if res:
        res_split = res.split(':')
        if len(res_split) > 1:
            try:
                for k in range(1, len(res_split)):
                    val = int(res_split[k])
            except ValueError:
                res = ''

    return res


def parse_instruction_str(s: str) -> "list[str]":
    """s a string, output from a neural net, intended to be a list of midi-like instructions where each is preceded
    by ;"""
    s_split = s.split(';')
    res = []
    for instruction in s_split:
        parsed = parse_single_instruction(instruction)
        if parsed:
            res.append(parsed)
    return res


# this one will be widely used
def instructions_by_extra_id(s: str) -> "collections.defaultdict[str, list[str]]":
    """s a string, output from a neural net, intended to be a list of midi-like instructions where each is preceded
    by ;

    returns a dict of the form
    <extra_id_n>: [x | x is an instruction following this extra_id in s and x is before the next extra_id in s].
    Note: Instructions in the values lists do NOT contain leading ;'s.
    Example: <extra_id_36>: ['D:36', 'D:49', 'w:12', 'D:42', 'w:12', 'D:38', 'D:42']

    Any instructions before the first <extra_id_n> in s are in the list whose key is ''. """
    res = collections.defaultdict(list)
    parsed = parse_instruction_str(s)
    cur_key = ''
    for p in parsed:
        if p[:10] == '<extra_id_':
            end_idx = p.find('>')
            cur_key = p[:end_idx + 1]

        else:
            res[cur_key].append(p)
    return res


def extract_extra_ids(s: str, L=None) -> "list[str]":
    """call this with L=None"""
    if L is None:
        L = []
    first_loc = s.find(';<extra_id_')
    if first_loc != -1:
        second_loc = s.find('>', first_loc)
        L.append(s[first_loc: second_loc + 1])
        return extract_extra_ids(s[second_loc + 1:], L)
    else:
        return L


def infos_by_extra_id(s: str = "") -> "dict[str, dict[str]]":
    """s a neural net INPUT. result is a dict with keys of the form <extra_id_n>. Says nothing about commands at end
    of s."""
    res = {}
    deconstructed_s = deconstructed_input_str(s)

    s_split = s.split(';')
    measure_i = -1
    BPM_level = -1
    vel_level = -1
    measure_len = -1

    cur_res = {'inst': None,
               'inst_rep': 0,
               "measure_index": measure_i,
               "measure_len": measure_len,
               "user_commands": [],
               "BPM_level": BPM_level,
               "vel_level": vel_level,
               }

    for instruction in s_split:
        if instruction[0: 2] == 'M:':
            measure_i += 1
            cur_res['measure_index'] = measure_i
            vel_level = int(instruction[2:])
            cur_res['vel_level'] = vel_level

            BPM_level = -1
            measure_len = -1

        elif instruction[0: 2] == 'B:':
            BPM_level = int(instruction[2:])
            cur_res['BPM_level'] = BPM_level
        elif instruction[0: 2] == 'L:':
            measure_len = int(instruction[2:])
            cur_res['measure_len'] = measure_len
        elif instruction[0: 2] == 'I:':
            cur_res['inst'] = int(instruction[2:])
            cur_res['inst_rep'] = 0
        elif instruction[0: 2] == 'R:':
            cur_res['inst_rep'] = int(instruction[2:])
        # elif instruction[0: 13] in ('<poly>', '<mono>', '<instruction_'):
        #     cur_res['user_commands'].append(instruction)
        elif instruction[0: 10] == '<extra_id_':
            user_commands = deconstructed_s[(cur_res["inst"], cur_res["inst_rep"], cur_res["measure_index"])]
            user_commands = user_commands.split(';')
            if len(user_commands) > 0:
                user_commands = user_commands[1:]
            while instruction in user_commands:
                user_commands.pop(user_commands.index(instruction))

            cur_res["user_commands"] = user_commands
            res[instruction] = cur_res
            cur_res = {'inst': None,
                       'inst_rep': 0,
                       "measure_index": measure_i,
                       "measure_len": measure_len,
                       "user_commands": [],
                       "BPM_level": BPM_level,
                       "vel_level": vel_level,
                       }

    return res


# test written
def deconstructed_input_str(s: str) -> "dict[tuple[int, int, int], str]":
    """s a neural net INPUT.

    res is a map of the form (inst, inst_rep, measure) -> whatever comes next for that inst: either ;<extra_...
    or e.g. ;d:12;N:36 etc. Ignores/removes commands at end.
    """
    res = {}

    sep_str = enc.instruction_str(0, enc.ENCODING_INSTRUCTION_INSTRUCTIONS_AT_END_SEP)
    sep_index = s.find(sep_str)
    if sep_index != -1:
        s = s[:sep_index]

    s_split = s.split(';')
    measure_i = -1

    cur_inst = -1

    cur_key = None
    cur_val = ''

    def update_res():
        if cur_key is not None and cur_val and cur_inst > -1:
            res[cur_key] = cur_val

    for instruction in s_split:
        if instruction[0: 2] == 'M:':
            measure_i += 1
        elif instruction[0: 2] == 'I:':
            cur_inst = int(instruction[2:])
            cur_inst_rep = 0
            update_res()
            cur_key = (cur_inst, cur_inst_rep, measure_i)
            cur_val = ''
        elif instruction[0: 2] == 'R:':
            cur_inst_rep = int(instruction[2:])
            update_res()
            cur_key = (cur_inst, cur_inst_rep, measure_i)
            cur_val = ''
        elif instruction[0: 2] == 'B:':
            pass
        elif instruction[0: 2] == 'L:':
            pass

        else:
            if instruction:
                cur_val += ';' + instruction

    update_res()
    return res
