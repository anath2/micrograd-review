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
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _grad():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _grad
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _grad():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        return Value(
            data=(self.data * other.data),
            _children=(self, other),
            _op='*',
            _backward=_grad,
        )

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, _children=(self, other), _op=f'**{other}')
        
        def _grad():
            self.grad += (
                other * self.data ** (other - 1)) * out.grad
        # Ignore other.grad since it's not used in the backward pass
        out._backward = _grad
        return out

    def relu(self):
        out = Value(
            0 if self.data < 0 else self.data,
            _children=(self,),
            _op='ReLU',
        )
        def _grad():
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _grad
        return out

    def backward(self):
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v):
            for child in v._prev:
                if child not in visited:
                    build_topo(child)
            topo.append(v)
            visited.add(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

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