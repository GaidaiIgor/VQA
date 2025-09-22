import time

import numpy as np
import qiskit
from networkx import Graph
from numpy import random
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_ionq import IonQProvider

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.PowerGrid import GeneratorCommitmentProblem, Generator, PowerFlowProblem
from src.Sampler import ExactSampler, MySamplerV2, IonQSampler
from src.VariationalCircuit import VariationalCircuit


def get_generator_commitment_problem() -> GeneratorCommitmentProblem:
    generators = np.array([Generator((15, 20), (0, 1, 10)),
                           Generator((0, 10), (1, 0, 1)),
                           Generator((0, 10), (2, 0, 1)),
                           Generator((0, 10), (3, 0, 1)),
                           Generator((0, 10), (4, 0, 1)),
                           Generator((0, 10), (5, 0, 1)),
                           Generator((0, 10), (6, 0, 1)),
                           Generator((0, 10), (7, 0, 1)),
                           Generator((0, 10), (8, 0, 1)),
                           Generator((0, 10), (9, 0, 1))])
    load = 10

    # generators = np.array([Generator((100, 600), (0.002, 10, 500)),
    #                        Generator((100, 400), (0.0025, 8, 300)),
    #                        Generator((50, 200), (0.005, 6, 100))])
    # load = 170
    #
    # generators = np.array([Generator((150, 455), (0.00048, 16.19, 1000)),
    #                        Generator((150, 455), (0.00031, 17.26, 970)),
    #                        Generator((20, 130), (0.002, 16.6, 700)),
    #                        Generator((20, 130), (0.00211, 16.5, 680)),
    #                        Generator((25, 162), (0.00398, 19.7, 450)),
    #                        Generator((20, 80), (0.00712, 22.26, 370)),
    #                        Generator((25, 85), (0.00079, 27.74, 480)),
    #                        Generator((10, 55), (0.00413, 25.92, 660)),
    #                        Generator((10, 55), (0.00222, 27.27, 665)),
    #                        Generator((10, 55), (0.00173, 27.79, 670))
    #                        ])
    # load = 700

    problem = GeneratorCommitmentProblem(generators, load)
    return problem


def get_power_flow_problem() -> PowerFlowProblem:
    graph = Graph()
    graph.add_node(0, generators=[Generator((0, 300), (0.01, 10, 10))], load=0)
    graph.add_node(1, generators=[Generator((0, 100), (0.02, 20, 1))], load=100)
    graph.add_node(2, generators=[], load=50)

    graph.add_edge(0, 1, capacity=100)
    graph.add_edge(0, 2, capacity=100)
    graph.add_edge(1, 2, capacity=100)
    return PowerFlowProblem(graph)


def get_job_cost(circuit: QuantumCircuit, nfev: int) -> tuple[int, int, int, float]:
    native_backend = IonQProvider().get_backend("qpu.forte-1", gateset="native")
    circuit_transpiled = qiskit.transpile(circuit, native_backend)
    num_one_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 1)
    num_two_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 2)
    return num_one_qubit_gates, num_two_qubit_gates, nfev, (num_one_qubit_gates * 0.0001645 + num_two_qubit_gates * 0.0011213) * nfev


def main():
    problem = get_generator_commitment_problem()
    # problem = get_power_flow_problem()
    num_gen = len(problem.generators)

    entangler = AllToAllEntangler(num_gen)
    mixer = ZXMixer(num_gen)
    num_layers = 1

    # sampler = ExactSampler()
    sampler = MySamplerV2(StatevectorSampler(default_shots=1))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-1", 1000, None)

    seed = 0
    rng = random.default_rng(seed)

    vqa = VariationalCircuit(num_layers, [entangler, mixer], sampler)
    initial_angles = rng.uniform(-np.pi, np.pi, len(vqa.circuit.parameters))
    result = vqa.optimize_parameters(problem.evaluate, initial_angles)

    best_sample = min(problem.optimize_power.cache.items(), key=lambda pair: pair[1].total)
    print(f"Angle optimization successful: {result.success}")
    print(f"State-average cost: {result.total}")
    print("=== Best sample ===")
    print(f"Power optimization successful: {best_sample[1].success}")
    print(f"Generators selected: {best_sample[0]}")
    print(f"Optimized power: {best_sample[1].x}")
    print(f"Optimized cost: {best_sample[1].fun}")
    print(f"Penalty: {best_sample[1].penalty}")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
