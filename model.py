import math
import pandas
import warnings

class Model:
    __state = []
    __probability = pandas.DataFrame()
    __tick = 0

    def step(self):
        if not self.__probability_check(strict = True):
            warnings.warn("Probability table check failed.")
        for index, row in self.__probability.mul([i._patient for i in self.__state], axis = 0).sum(axis = 0).iteritems():
            state = self.state(index)
            if not state._patient.callable():
                state.patient = row
        self.__tick += 1
    
    def __state_index(self, v):
        if isinstance(v, State):
            return self.__state.index(v)
        else:
            return [s.name for s in self.__state].index(v)
    
    def _generator(self, row):
        t = self.__probability.loc[row]
        value = 1
        for i in t[[not i.callable() for i in t]]:
            value -= i.value
        if value < 0 or value > 1:
            raise ValueError("ValueError: probablitiy should be between 0 to 1")
        return value
    
    def state(self, name, utility: float = None, cost: float = None, patient: float = None, strict: bool = False):
        index = None
        try:
            index = self.__state_index(name)
        except ValueError:
            if type(name) != str:
                raise TypeError("TypeError: state name should be string")
            if strict:
                raise ValueError("ValueError: '{}' is not in list".format(name))
            s = State(name, model = self)
            self.__state.append(s)
            index = -1
            
            self.__probability[name] = Generator(0)
            self.__probability.loc[name] = Generator(0)
            self.__probability.at[name, name] = Generator(s.generator)
        
        if utility:
            self.__state[index].utility = utility
        if cost:
            self.__state[index].cost = cost
        if patient:
            self.__state[index].patient = patient
            
        return self.__state[index]

    def transit(self, start, end, probability):
        start = start if type(start) == str else start.name
        end = end if type(end) == str else end.name
        self.__probability.at[start, end] = Generator(probability)
        if not self.__probability_check():
            warnings.warn("Probability table check failed.")
        
    def __probability_check(self, strict: bool = False):
        if self.__probability is None:
            return True
        if (self.__probability < 0).any().any():
            return False
        for _, row in self.__probability.iterrows():
            sums = 0
            for i in row:
                sums += i.value
            if strict and not math.isclose(sums, 1):
                return False
            elif not strict and sums > 1:
                return False
        return True
        
    def patient(self, state, value: float):
        self.state(state, patient = value, strict = True)
    
    def utility(self, state, value: float):
        self.state(state, utility = value, strict = True)

    def cost(self, state, value: float):
        self.state(state, cost = value, strict = True)

    @property
    def patients(self):
        return sum([state.patient for state in self.__state])
    
    @property
    def utilities(self):
        return sum([state.utilities for state in self.__state])

    @property
    def costs(self):
        return sum([state.costs for state in self.__state])
    
    @property
    def tick(self):
        return self.__tick

    @property
    def probability(self):
        return self.__probability
    
    def information(self):
        print("==========states==========")
        print(" | ".join([i.name for i in self.__state]) + " [total" + "]")
        print("==========utilities==========")
        print(" | ".join(["{:.2f}".format(i.utility) for i in self.__state]) + " [{:.2f}".format(self.utilities) + "]")
        print("==========costs==========")
        print(" | ".join(["{:.2f}".format(i.cost) for i in self.__state]) + " [{:.2f}".format(self.costs) + "]")
        print("==========patients==========")
        print(" | ".join(["{:.2f}".format(i.patient) for i in self.__state]) + " [{:.2f}".format(self.patients) + "]")

class Generator:
    _value = 0
    _generator = {}

    def __init__(self, value = 0):
        self.value = value

    @property
    def value(self):
        return self._value(**self._generator) if self.callable() else self._value
        
    @value.setter
    def value(self, value: float = 1):
        self._value = value

    def __repr__(self):
        return self.value.__repr__()
        
    def generator(self, **value):
        self._generator.update(value)

    def callable(self):
        return callable(self._value)
    
    def __eq__(self, other):
        return self.value.__eq__(other.value if type(other) == Generator else other)
    
    def __ne__(self, other):
        return self.value.__ne__(other.value if type(other) == Generator else other)

    def __lt__(self, other):
        return self.value.__lt__(other.value if type(other) == Generator else other)
    
    def __le__(self, other):
        return self.value.__le__(other.value if type(other) == Generator else other)
    
    def __gt__(self, other):
        return self.value.__gt__(other.value if type(other) == Generator else other)
    
    def __ge__(self, other):
        return self.value.__ge__(other.value if type(other) == Generator else other)
    
    def __bool__(self):
        return self.value.__bool__()
    
    def __add__(self, other):
        return self.value + (other.value if type(other) == Generator else other)
    
    def __sub__(self, other):
        return self.value - (other.value if type(other) == Generator else other)
    
    def __mul__(self, other):
        return self.value * (other.value if type(other) == Generator else other)
            
class State:
    name = None
    _patient = None
    _utility = None
    _cost = None
    _model = None

    def __init__(self, name: str, patient = 0, utility = 1, cost = 0, model: Model = None):
        assert (patient if patient else 0) >= 0, "Patient cannot be negative."

        self.name = name
        self.patient = patient
        self.utility = utility
        self.cost = cost
        self._model = model

    @property
    def patient(self):
        return self._patient.value
    
    @patient.setter
    def patient(self, value = 0):
        if not self._patient:
            self._patient = Generator(value)
        else:
            self._patient.value = value

    @property
    def utility(self):
        return self._utility.value
    
    @utility.setter
    def utility(self, value = 1):
        if not self._utility:
            self._utility = Generator(value)
        else:
            self._utility.value = value

    @property
    def cost(self):
        return self._cost.value
    
    @cost.setter
    def cost(self, value = 0):
        if not self._cost:
            self._cost = Generator(value)
        else:
            self._cost.value = value

    def transit(self, state, probability):
        self._model.transit(self, state, probability)

    @property
    def utilities(self):
        return self._utility.value * self._patient.value
    
    @property
    def costs(self):
        return self._cost.value * self._patient.value
    
    def generator(self):
        return self._model._generator(self.name)