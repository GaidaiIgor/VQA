import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


@dataclass
class CircuitLayer(ABC):
    """
    A layer of a VQA.
    :var num_qubits: Number of qubits in the circuit.
    """
    num_qubits: int

    @staticmethod
    def connect_pair(qc: QuantumCircuit, i: int, j: int, angle: Parameter):
        """ Appends a couping gate with the given parameter between qubits i and j to the given quantum circuit. """
        qc.rzz(angle, i, j)

    @abstractmethod
    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        """ Returns qiskit circuit for the layer. Appends name_suffix to the variable names. """
        pass


class AllToAllEntangler(CircuitLayer):
    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector(f"G_{name_suffix}", self.num_qubits * (self.num_qubits - 1) // 2)
        for ind, (i, j) in enumerate(combinations(range(self.num_qubits), 2)):
            self.connect_pair(qc, i, j, params[ind])
        return qc


class ButterflyEntangler(CircuitLayer):
    def connect_qubits(self, qc: QuantumCircuit, qubit_range: tuple[int, int], name_suffix: str):
        """ Applies couplings in the specified qubit range. """
        range_len = qubit_range[1] - qubit_range[0]
        if range_len < 2:
            return
        step = 2 ** (math.ceil(math.log2(range_len)) - 1)
        params = ParameterVector(f"G_{name_suffix}_{qubit_range[0]}{qubit_range[1]}", range_len - step)
        for ind, start in enumerate(range(qubit_range[0], qubit_range[0] + len(params))):
            self.connect_pair(qc, start, start + step, params[ind])
        self.connect_qubits(qc, (qubit_range[0], sum(qubit_range) // 2), name_suffix)
        self.connect_qubits(qc, (sum(qubit_range) // 2, qubit_range[1]), name_suffix)

    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        self.connect_qubits(qc, (0, self.num_qubits), name_suffix)
        return qc


class ZXMixer(CircuitLayer):
    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector(f"B_{name_suffix}", 2 * self.num_qubits)
        for i in range(self.num_qubits):
            qc.rz(params[i], i)
        for i in range(self.num_qubits):
            qc.rx(params[self.num_qubits + i], i)
        return qc
