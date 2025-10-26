from typing import Any


class Value:
    def __init__(self, data: Any, label: str = '', _children: tuple = (), _op: str = ''):
        self.grad = 0
        self.data = data
        self.label = label
        self._children = set(_children)
        self._op = _op

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        return Value(data=(self.data + other.data), _children=(self, other), _op='+')

    def __mul__(self, other):
        return Value(data=(self.data * other.data), _children=(self, other), _op='*')
