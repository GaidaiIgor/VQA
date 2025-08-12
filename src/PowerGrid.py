from dataclasses import dataclass, field

from numpy import ndarray
from numpy import random
from scipy import optimize
from scipy.optimize import OptimizeResult


@dataclass
class Generator:
    """
    Describes a generator
    :var power_range: (min, max) values of power for this generator.
    :var cost_terms: (a, b, c) terms of quadratic generation cost function (ap^2 + bp + c).
    """
    power_range: tuple[float, float]
    cost_terms: tuple[float, float, float]

    def generation_cost(self, power: float) -> float:
        return self.cost_terms[0] * power ** 2 + self.cost_terms[1] * power + self.cost_terms[2]


@dataclass
class PowerGridUnitCommitmentProblem:
    """ Describes a unit commitment problem in power grid.
    I.e. given a set of generators, which ones should be enabled and at what power in order to meet target load using the smallest operation cost. """
    generators: ndarray[Generator]
    load: float


@dataclass
class CostFunction:
    """ Describes power grid cost function. Evaluates and stores known results. """
    problem: PowerGridUnitCommitmentProblem
    known_values: dict[str, OptimizeResult] = field(default_factory=dict)

    def optimize_power(self, bitstring: str) -> OptimizeResult:
        def generation_cost(powers: list[float]) -> float:
            return sum(gen.generation_cost(power) for gen, power in zip(enabled_generators, powers))

        def generation_cost_total(powers: list[float]) -> float:
            cost = generation_cost(powers)
            constraint_dist = constraint["fun"](powers)
            if constraint_dist < 0:
                penalty = 1e2 * constraint_dist ** 2
                cost += penalty
            return cost

        enabled_generators = self.problem.generators[[int(val) == 1 for val in bitstring]]
        initial_point = [random.uniform(*gen.power_range) for gen in enabled_generators]
        bounds = [gen.power_range for gen in enabled_generators] if enabled_generators.size > 0 else None
        constraint = {"type": "ineq", "fun": lambda powers: sum(powers) - self.problem.load}
        result = optimize.minimize(generation_cost_total, initial_point, method="SLSQP", bounds=bounds, constraints=constraint)
        result.generation_cost = generation_cost(result.x)
        result.constraint = constraint["fun"](result.x)
        return result

    def evaluate(self, bitstring: str) -> float:
        if bitstring not in self.known_values:
            self.known_values[bitstring] = self.optimize_power(bitstring)
        return self.known_values[bitstring].fun


