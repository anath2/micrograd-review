from typing import Any


class Value:
    def __init__(self, data: Any, label: str = '', _children: tuple = (), _op: str = ''):
        self.data = data
        self.grad = 0
        # internal variables for autograd graph construction
        self._backward = lambda: None
        self._prev= set(_children)
        self._op = _op  # operation that produced this value

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        def _grad():
            return 1.0

        return Value(
            data=(self.data + other.data),
            _children=(self, other),
            _op='+',
            _backward=_grad,
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        def _grad():
            return other.data
        
        return Value(
            data=(self.data * other.data),
            _children=(self, other),
            _op='*',
            _backward=_grad,
        )

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        def _grad():
            return other.data * (self.data ** (other.data - 1))
        
        return Value(
            data=(self.data ** other.data),
            _children=(self, other),
            _op=f'**{other}',
            _backward=_grad,
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'