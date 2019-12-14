import numpy as np
from typing import List, Tuple
from random import random, randint, shuffle, choice
from functools import reduce
import abc

eta = .03
reps = 1000

def parse_example(line: str) -> Tuple[List[float], List[float]]:
    words = line.split()
    return [float(words[0]), float(words[1])], [float(words[2]) - 1]

def parse_file(path: str) -> List[Tuple[List[float], List[float]]]:
    with open(path, "r") as f:
        return [parse_example(line) for line in f]

def sigmoid(x: float) -> float:
    return 1.0/(1+ np.exp(-x))

def d_sigmoid(x: float) -> float:
    return x * (1.0 - x)

def d_tanh(x: float) -> float:
    return 1 - np.tanh(x)**2

class Layer(abc.ABC):
    def output_values(self, inputs: List[float]) -> List[float]:
        pass

    def backprop(self, errors: List[float]) -> List[float]:
        pass

class SigmoidLayer(Layer):
    def __init__(self, size: int):
        self.outputs = [0.]*size
        self.input_errors = [0.]*size

    def output_values(self, inputs: List[float]) -> List[float]:
        self.outputs = list (map (sigmoid, inputs))
        return self.outputs

    def backprop(self, errors: List[float]) -> List[float]:
        self.input_errors = [d_sigmoid(o) * e for (e, o) in zip(errors, self.outputs)]
        return self.input_errors

class LinearCompleteLayer(Layer):
    def __init__(self, in_size: int, out_size: int):
        self.outputs = [0.]*out_size
        self.inputs: List[float] = []
        self.input_errors = [0.]*(in_size + 1)
        self.weights:List[List[float]] = [[choice([-.5,.5]) * random() for _ in range(in_size + 1)] for _ in range(out_size)]

    def output_values(self, inputs: List[float]) -> List[float]:
        self.inputs = inputs + [1]
        self.outputs = [sum ([w*i for (w, i) in zip (ws, self.inputs)]) for ws in self.weights]
        return self.outputs

    def backprop(self, errors: List[float]) -> List[float]:
        for j in range(len(self.weights)): # ws is the list of weights for each output
            for i in range(len(self.weights[j])):
                self.weights[j][i] += eta * self.inputs[i] * errors[j]
        self.input_errors = []
        for i in range(len(self.weights[0])):
            wt_sum = 0.
            for j, ws in enumerate(self.weights):
                wt_sum += ws[i] * errors[j]
            self.input_errors.append(wt_sum)
        return self.input_errors

def sum_sq_error_layer(ys, ycaps):
    return [pow((y - yc),1) for (y, yc) in zip (ys, ycaps)]

def sumsqerr(ys, ycaps):
    return sum([pow((y - yc),2) for (y, yc) in zip (ys, ycaps)])

def train_classifier(network, examples, reps=100):
    for _ in range(reps):
        shuffle(examples)
        for e in examples:
            values = e[0]
            for layer in layers:
                values = layer.output_values(values)
            error = sum_sq_error_layer(e[1], values)
            for layer in reversed(layers):
                error = layer.backprop(error)
        reps += 1

def weights_to_string(weights: List[List[float]]) -> str:
    return reduce(lambda acc, xs: acc + xs + "\n", map(lambda ws: "\t".join(map(lambda w: str(w), ws)) ,weights), "")

def print_results(path, layers, eta, reps, err):
    with open(path, "w") as f:
        f.write("CS-5001: HW#3\n")
        f.write("Programmer: Drew Willey\n")
        f.write("\n")
        f.write("TRAINING\n")
        f.write(f"Using learning rate eta = {eta}\n")
        f.write(f"Using {reps} iterations\n")
        f.write("\n")
        f.write("OUTPUT:\n")
        f.write("Input Units:\n")
        f.write(weights_to_string(layers[0].weights))
        f.write("Output Unit:\n")
        f.write(weights_to_string(layers[2].weights))
        f.write("\n")
        f.write("VALIDATION\n")
        f.write(f"Sum-of-Squares Error = {err}")

def run_classifier(layers, examples):
    ys = []
    ycs = []
    for e in examples:
        values = e[0]
        for layer in layers:
            values = layer.output_values(values)
        ys += e[1]
        ycs += values
        print(e[0], values, e[1], sum_sq_error_layer(e[1], values))
    return ys, ycs

if __name__ == "__main__":
    layers: List[Layer] = [LinearCompleteLayer(2,6), SigmoidLayer(6), LinearCompleteLayer(6,1), SigmoidLayer(1)]
    examples = parse_file("hw3data.txt")
    val_examples = parse_file("hw3valid.txt")
    train_classifier(layers, examples, reps)
    ys, ycs = run_classifier(layers, val_examples)
    print_results("classifieroutput.txt", layers, eta, reps, sumsqerr(ys, ycs))