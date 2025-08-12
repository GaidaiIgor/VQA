from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Statevector


class Sampler(ABC):
    """ Base class for samplers. """

    @abstractmethod
    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """ Assigns given values to given parameterized circuit, executes it and returns dictionary where keys are bitstrings are values are their sampling probabilities. """
        pass


class ExactSampler(Sampler):
    """ Calculates exact probabilities of each bitstring. """

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        bound = circuit.assign_parameters(param_vals)
        return Statevector(bound).probabilities_dict()


@dataclass
class MySamplerV2(Sampler):
    """ Uses sampler compatible with BaseSamplerV2 interface. """
    sampler: BaseSamplerV2

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        measurement_circuit = circuit.measure_all(inplace=False)
        counts = self.sampler.run([(measurement_circuit, param_vals)]).result()[0].data.meas.get_counts()
        probabilities = {key: value / self.sampler.default_shots for key, value in counts.items()}
        return probabilities
