from abc import abstractmethod

##################
# Basic containers


# test written
class BasicContainer(object):
    @abstractmethod
    def __getstate__(self):
        raise NotImplemented('inheriting class must define __getstate__')

    @abstractmethod
    def __copy__(self):
        raise NotImplemented('inheriting class must define __copy__')

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__getstate__() == other.__getstate__()

    def __ne__(self, other):
        return not self == other


# test written
class TimeSig(BasicContainer):
    __slots__ = ("_num", "_denom", "_click")

    def __getstate__(self):
        return self.num, self.denom, self.click

    def __copy__(self):
        return self.__class__(num=self.num, denom=self.denom, click=self.click)

    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, val):
        if val < 1:
            raise ValueError('Time signature numerator < 1: {}'.format(val))
        self._num = val

    @property
    def denom(self):
        return self._denom

    @denom.setter
    def denom(self, val):
        if val not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            raise ValueError('Time signature denominator invalid: {}'.format(val))
        self._denom = val

    @property
    def click(self):
        return self._click

    @click.setter
    def click(self, val):
        if val < 0:
            raise ValueError('Click {} is negative'.format(val))
        self._click = int(val)

    def __init__(self, num=4, denom=4, click=0):
        self.num = num
        self.denom = denom
        self.click = click

    def __repr__(self):
        return "(TS:{}/{} c={})".format(self.num, self.denom, self.click)

    def is_equiv_to(self, other):
        return self.num * other.denom == other.num * self.denom

    @classmethod
    def from_miditoolkit_TimeSignature(cls, X):
        return cls(num=int(X.numerator), denom=int(X.denominator), click=int(X.time))


class Marker(BasicContainer):
    __slots__ = ('text', 'click')

    def __getstate__(self):
        return self.text, self.click

    def __copy__(self):
        return self.__class__(text=self.text, click=self.click)

    def __init__(self, text='', click=0):
        self.text = text
        self.click = click

    def __repr__(self):
        return "(Marker:{}:{})".format(self.text, self.click)

    @classmethod
    def from_miditoolkit_Marker(cls, X):
        return cls(text=str(X.text), click=int(X.time))


# test written
class Note(BasicContainer):
    __slots__ = ("pitch", "_vel", "_click", "_end", "inst")

    def __getstate__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        return self.pitch, self.vel, self.click, self.end, inst

    def __copy__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        return self.__class__(pitch=self.pitch, vel=self.vel, click=self.click, end=self.end, inst=inst)

    @property
    def click(self):
        return self._click

    @click.setter
    def click(self, val):
        self._click = int(val)
        if val > self.end:
            self.end = val

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        self._end = int(val)
        if val < self.click:
            self._click = val

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, val):
        if val < 1:
            val = 1
        if val > 127:
            val = 127
        self._vel = int(val)

    def __init__(self, pitch=0, vel=1, click=0, end=None, inst=None):
        """This class represents a note on followed by a note off. The note on occurs at the .click value, and the
        note off occurs at the .end value. Both must be supplied. If end is not supplied or is
        prior to click, then end is set to click.
        Instrument can be an attribute of a note, but more commonly is an attribute of the track containing the note.
        Notes are guaranteed to have .pitch, .vel, .click, and .end values. May or may not have .inst attribute.
        """
        self.pitch = int(pitch)

        if inst is not None:
            self.inst = int(inst)

        # these use the setters
        if end is None:
            end = click
        # little hack since click setter looks at end, and end setter looks at click
        self._click = 0
        self._end = 0

        self.vel = vel
        self.end = end
        self.click = click

    @classmethod
    def from_miditoolkit_Note(cls, X):
        return cls(pitch=int(X.pitch), vel=int(X.velocity), click=int(X.start), end=int(X.end))

    def __repr__(self):
        res = []
        res.append('N:{}'.format(self.pitch))
        res.append('v:{}'.format(self.vel))
        if hasattr(self, "inst"):
            res.append('inst:{}'.format(self.inst))
        res.append("click:{}".format(self.click))
        if self.end > self.click:
            res.append("end:{}".format(self.end))

        if res:
            return '(' + ' '.join(res) + ')'
        else:
            return ''


class NoteOn(BasicContainer):
    __slots__ = ('pitch', 'vel', 'click', 'inst', 'noteidx')

    def __getstate__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        noteidx = self.noteidx if hasattr(self, 'noteidx') else None
        return self.pitch, self.vel, self.click, inst, noteidx

    def __copy__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        noteidx = self.noteidx if hasattr(self, 'noteidx') else None
        return self.__class__(pitch=self.pitch, vel=self.vel, click=self.click, inst=inst, noteidx=noteidx)

    def __init__(self, pitch=0, vel=1, click=0, inst=None, noteidx=None):
        self.pitch = pitch
        self.vel = vel
        self.click = click
        if inst is not None:
            self.inst = inst
        if noteidx is not None:
            self.noteidx = noteidx

    def __repr__(self):
        res = []
        res.append('NOn:{}'.format(self.pitch))
        res.append('v:{}'.format(self.vel))
        if hasattr(self, 'inst'):
            res.append('inst:{}'.format(self.inst))
        res.append('click:{}'.format(self.click))
        if hasattr(self, 'noteidx'):
            res.append('idx:{}'.format(self.noteidx))
        if res:
            return '(' + ' '.join(res) + ')'
        else:
            return ''


class NoteOff(BasicContainer):
    __slots__ = ('pitch', 'click', 'inst', 'noteidx')

    def __getstate__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        noteidx = self.noteidx if hasattr(self, 'noteidx') else None
        return self.pitch, self.click, inst, noteidx

    def __copy__(self):
        inst = self.inst if hasattr(self, 'inst') else None
        noteidx = self.noteidx if hasattr(self, 'noteidx') else None
        return self.__class__(pitch=self.pitch, click=self.click, inst=inst, noteidx=noteidx)

    def __init__(self, pitch=0, click=0, inst=None, noteidx=None):
        self.pitch = pitch
        self.click = click
        if inst is not None:
            self.inst = inst
        if noteidx is not None:
            self.noteidx = noteidx

    def __repr__(self):
        res = []
        res.append('NOff:{}'.format(self.pitch))
        if hasattr(self, 'inst'):
            res.append('inst:{}'.format(self.inst))
        res.append('click:{}'.format(self.click))
        if hasattr(self, 'noteidx'):
            res.append('idx:{}'.format(self.noteidx))
        if res:
            return '(' + ' '.join(res) + ')'
        else:
            return ''


class CC(BasicContainer):
    __slots__ = ("cc", "val", "click")

    def __getstate__(self):
        return self.cc, self.val, self.click

    def __copy__(self):
        return self.__class__(cc=self.cc, val=self.val, click=self.click)

    def __init__(self, cc=0, val=0, click=0):
        self.cc = cc
        self.val = val
        self.click = click

    @classmethod
    def from_miditoolkit_ControlChange(cls, X):
        return cls(click=int(X.time), val=int(X.value), cc=int(X.number))

    def __repr__(self):
        res = '(cc:{} v:{} click:{})'.format(self.cc, self.val, self.click)
        return res


# test written
class Pedal(BasicContainer):
    __slots__ = ("_click", "_end")

    def __getstate__(self):
        return self.click, self.end

    def __copy__(self):
        return self.__class__(click=self.click, end=self.end)

    @property
    def click(self):
        return self._click

    @click.setter
    def click(self, val):
        self._click = int(val)
        if val > self.end:
            self.end = val

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        self._end = int(val)
        if val < self.click:
            self._click = val

    def __init__(self, click=0, end=0):
        # little hack since click setter looks at end, and end setter looks at click
        self._click = 0
        self._end = 0

        self.end = end
        self.click = click

    @classmethod
    def from_miditoolkit_Pedal(cls, X):
        return cls(click=int(X.start), end=int(X.end))

    def __repr__(self):
        res = '(Pedal click:{} end:{})'.format(self.click, self.end)
        return res


class PitchBend(BasicContainer):
    __slots__ = ("val", "click")

    def __getstate__(self):
        return self.val, self.click

    def __copy__(self):
        return self.__class__(val=self.val, click=self.click)

    def __init__(self, val=0, click=0):
        self.val = val
        self.click = click

    @classmethod
    def from_miditoolkit_PitchBend(cls, X):
        return cls(val=int(X.pitch), click=int(X.time))

    def __repr__(self):
        res = '(PB v:{} click:{})'.format(self.val, self.click)
        return res


class TempoChange(BasicContainer):
    __slots__ = ("val", "click")

    def __getstate__(self):
        return self.val, self.click

    def __copy__(self):
        return self.__class__(val=self.val, click=int(self.click))

    def __init__(self, val, click=0):
        if val <= 0:
            raise ValueError('TempoChange val (BPM) must be > 0')
        self.val = val
        self.click = click

    @classmethod
    def from_miditoolkit_TempoChange(cls, X):
        return cls(val=float(X.tempo), click=int(X.time))

    def __repr__(self):
        res = '(Tempo v:{} click:{})'.format(self.val, self.click)
        return res


class NoteIdxInfo(object):
    def __init__(self, note_on, measure_note_on, note_off=None, measure_note_off=None):
        """
        :type note_on: NoteOn
        :type measure_note_on: int
        :type note_off: NoteOff or None
        :type measure_note_off: int or None
        """
        self.note_on = note_on
        self.note_off = note_off
        self.measure_note_on = measure_note_on
        self.measure_note_off = measure_note_off
        self.length = None

    def __repr__(self):
        return 'NoteIdxInfo:' + str(self.__dict__)
