import bisect
import sys


def tostr(L):
    res = [str(x) for x in L]
    return res


# this actually needs to be defined in the same reaper py file it is used in. It's just here for quick and easy copying.
def _patch_stdout_stderr_open():
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


def index_of_closest_element_in_sorted_numeric_list(L, x):
    '''returns i such that abs(x - L[i]) is minimized. If there is a tie, defaults to the rightmost index.'''
    i = bisect.bisect_right(L, x)
    if i == 0:
        return i
    elif i == len(L):
        return i-1
    elif L[i-1] == x:
        return i-1
    else:
        a, b = L[i-1], L[i]
        if x-a < b-x:
            return i-1
        else:
            return i


def adjacent_elts(L, idx):
    '''L a nonempty list, 0<=idx<len(L). Returns the elts of L on either side of the index idx.'''
    if len(L) == 0:
        raise ValueError('list is empty')
    if idx < 0 or idx >= len(L):
        raise ValueError('idx is out of range')

    if len(L) == 1:
        return []

    # at this point we know len(L) >= 2
    if idx == 0:
        return [L[1]]

    if idx == len(L) - 1:
        return [L[idx - 1]]

    return [L[idx - 1], L[idx + 1]]


def iter_adjacent_pairs(iterable):
    """if iterable = [1, 2, 3, 4, 5], outputs (1,2), (2,3), (3,4), (4,5)"""
    for i, x in enumerate(iterable):
        try:
            yield x, iterable[i+1]
        except IndexError:
            pass


def iter_adjacent_triples(iterable):
    """if iterable = [1, 2, 3, 4, 5], outputs (1,2, 3), (2, 3, 4), (3, 4, 5)"""
    for i, x in enumerate(iterable):
        try:
            yield x, iterable[i+1], iterable[i+2]
        except IndexError:
            pass


def iter_adjacent_k_tuples(iterable, k=2):
    if k > 0:
        for i, x in enumerate(iterable):
            try:
                res = []
                for j in range(k):
                    res.append(iterable[i+j])
                yield res
            except IndexError:
                pass


def contains_at_least_two_elements(L):
    """L an iterable"""
    for i, x in enumerate(L):
        if i == 1:
            return True
    return False


def contains_exactly_one_element(L):
    """L an iterable"""
    if not L:
        return False
    for i, x in enumerate(L):
        if i > 0:
            return False
    return True


def fill_list_with_midpoints(L):
    """L a sorted list. Returns a new list."""
    if contains_exactly_one_element(L):
        return [L[0]]

    else:
        res = []

        if contains_at_least_two_elements(L):
            for P in iter_adjacent_pairs(L):
                res.append(P[0])
                res.append((P[0]+P[1]) / 2.0)
            res.append(P[1])
        return res


def merge_list_of_dictionaries(L):
    """L a list of dictionaries.
    Returns a new dictionary that merges all dictionaries in L.
    Keys in later dicts override those of earlier dicts."""
    res = {}
    for d in L:
        res.update(d)
    return res


def is_approx(x, y, tol=0.0001):
    """return abs(x-y) <= tol"""
    return abs(x-y) <= tol
