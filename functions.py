"""Functions for stuff"""
from dataclasses import dataclass
from typing import Callable
import math


def relu(x: float) -> float:
    return max(0.0, x)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def tanh(x: float) -> float:
    return math.tanh(x)


@dataclass(frozen=True)
class ActivationFunction:
    """
    Immutable data class for activation functions.

    An activation function determines the output of a neural network node
    given an input or set of inputs. This dataclass provides a standardized
    interface for different activation functions.

    Attributes:
        func: A callable that takes a float input and returns a float output.
              Common examples include ReLU, tanh, sigmoid.
    """
    name: str
    func: Callable[[float], float]

RELU = ActivationFunction("relu", relu)
SIGMOID = ActivationFunction("sigmoid", sigmoid)
TANH = ActivationFunction("tanh", tanh)
