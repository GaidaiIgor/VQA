from typing import Callable, Sequence

import noisyopt
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from scipy import optimize
from scipy.optimize import OptimizeResult

from src.CircuitLayer import CircuitLayer
from src.Sampler import Sampler, ExactSampler


class VariationalCircuit:
    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.layer_types[0].num_qubits)
        qc.h(range(qc.num_qubits))
        # qc.barrier()
        for i in range(self.num_layers):
            for layer_type in self.layer_types:
                qc.compose(layer_type.get_circuit(str(i)), inplace=True)
                # qc.barrier()
        return qc

    def __init__(self, num_layers: int, layer_types: list[CircuitLayer], sampler: Sampler):
        """ Appends specified number of layers. Each full layer is a combination of all circuits from layer_types. """
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.sampler = sampler
        self.circuit = self.build()

    def get_cost_expectation(self, cost_function: Callable[[str], float], param_vals: Sequence[float]):
        """ Evaluates expectation of the cost function at given circuit parameter values. """
        probabilities = self.sampler.get_sample_probabilities(self.circuit, param_vals)
        expectation = sum(cost_function(bitstring) * probability for bitstring, probability in probabilities.items())
        return expectation

    def optimize_parameters(self, cost_function: Callable[[str], float], initial_angles: ndarray) -> OptimizeResult:
        """ Optimizes variational parameters of the circuit to minimize expectation of cost function and returns optimized parameter values. """
        min_func = lambda angles: self.get_cost_expectation(cost_function, angles)
        if isinstance(self.sampler, ExactSampler):
            result = optimize.minimize(min_func, initial_angles, method="SLSQP", options={"maxiter": np.iinfo(np.int32).max})
        else:
            result = noisyopt.minimizeCompass(min_func, initial_angles, errorcontrol=False)
        return result
