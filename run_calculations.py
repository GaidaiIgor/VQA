import time

import numpy as np
from numpy import random
from qiskit.primitives import StatevectorSampler

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.PowerGrid import GeneratorCommitmentProblem, Generator, CostFunction
from src.Sampler import ExactSampler, MySamplerV2
from src.VariationalCircuit import VariationalCircuit


def get_problem() -> GeneratorCommitmentProblem:
    generators = np.array([Generator((15, 20), (0, 1, 10)),
                           Generator((0, 10), (1, 0, 1))])
    load = 10

    # generators = np.array([Generator((100, 600), (0.002, 10, 500)),
    #                        Generator((100, 400), (0.0025, 8, 300)),
    #                        Generator((50, 200), (0.005, 6, 100))])
    # load = 170

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


def main():
    problem = get_problem()
    num_gen = len(problem.generators)
    cost_function = CostFunction(problem)

    entangler = AllToAllEntangler(num_gen)
    mixer = ZXMixer(num_gen)
    sampler = ExactSampler()
    # sampler = MySamplerV2(StatevectorSampler())
    num_layers = 1

    vqa = VariationalCircuit(num_layers, [entangler, mixer], sampler)
    initial_angles = random.uniform(-np.pi, np.pi, len(vqa.circuit.parameters))
    result = vqa.optimize_parameters(cost_function.evaluate, initial_angles)

    best_sample = min(cost_function.known_values.items(), key=lambda pair: pair[1].fun)
    print(f"Angle optimization successful: {result.success}")
    print(f"State-average cost: {result.fun}")
    print("=== Best sample ===")
    print(f"Power optimization successful: {best_sample[1].success}")
    print(f"Generators selected: {best_sample[0]}")
    print(f"Optimized power: {best_sample[1].x}")
    print(f"Optimized cost: {best_sample[1].generation_cost}")
    print(f"Constraint: {best_sample[1].constraint}")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
