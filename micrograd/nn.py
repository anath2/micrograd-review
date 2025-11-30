import random
from micrograd.engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):

    def __init__(self, nin: int, non_linear: bool = True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.non_linear = non_linear

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.non_linear else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.non_linear else 'Linear'} Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x) -> Value | list[Value]:
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(map(repr, self.neurons))}]"
    

class MLP(Module):

    def __init__(self, nin: int, nouts: list[int], **kwargs) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], non_linear=i != len(sz) - 1) for i in range(len(sz) - 1)]

    def __call__(self, x) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(map(repr, self.layers))}]"
