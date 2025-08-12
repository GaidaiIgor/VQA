import numpy as np
from numpy import random
from qiskit.primitives import StatevectorSampler

from src.CircuitLayer import AllToAllEntangler, XMixer
from src.PowerGrid import PowerGridUnitCommitmentProblem, Generator, CostFunction
from src.Sampler import ExactSampler, MySamplerV2
from src.VariationalCircuit import VariationalCircuit


def get_problem() -> PowerGridUnitCommitmentProblem:
    generators = np.array([Generator((15, 20), (0, 0.1, 100)),
                           Generator((0, 10), (100, 0, 1))])
    load = 10
    problem = PowerGridUnitCommitmentProblem(generators, load)
    return problem


def main():
    problem = get_problem()
    num_gen = len(problem.generators)
    cost_function = CostFunction(problem)

    entangler = AllToAllEntangler(num_gen)
    mixer = XMixer(num_gen)
    sampler = ExactSampler()
    # sampler = MySamplerV2(StatevectorSampler())
    num_layers = 1

    vqa = VariationalCircuit(num_layers, [entangler, mixer], sampler)
    initial_angles = random.uniform(-np.pi, np.pi, len(vqa.circuit.parameters))
    result = vqa.optimize_parameters(cost_function.evaluate, initial_angles)

    best_sample = min(cost_function.known_values.items(), key=lambda pair: pair[1].fun)
    print(f"Angle optimization successful: {result.success}")
    print(f"Power optimization successful: {best_sample[1].success}")
    print(f"Constraint: {best_sample[1].constraint}")
    print(f"Generators selected: {best_sample[0]}")
    print(f"Optimized power: {best_sample[1].x}")
    print(f"Optimized cost: {best_sample[1].fun}")


if __name__ == "__main__":
    main()
