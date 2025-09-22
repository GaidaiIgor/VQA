from dataclasses import dataclass
from functools import wraps, partial
from typing import Callable, Hashable, Any

import networkx as nx
import numpy as np
from networkx import Graph
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import OptimizeResult


def cached[K: Hashable, T](func: Callable[[K], T]) -> Callable[[K], T]:
    """ Function decorator that saves evaluation results in function's internal dict and reads them if the function is called again with the same argument. """
    @wraps(func)
    def wrapper(arg: K) -> T:
        if arg not in wrapper.cache:
            wrapper.cache[arg] = func(arg)
        return wrapper.cache[arg]

    wrapper.cache = {}
    return wrapper


def get_penalty(powers: list[float], constraints: list[dict[str, Any]], mult: float = 1e1) -> float:
    """ Evaluates penalty term for a given power vector and list of constraints. """
    penalty = 0
    for constraint in constraints:
        val = constraint["fun"](powers)
        if val < 0:
            penalty += mult * val ** 2
    return penalty


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
class GeneratorCommitmentProblem:
    """ Describes a unit commitment problem in a power grid.
    I.e. given a set of generators, which ones should be enabled and at what power in order to meet target load using the smallest operation cost. """
    generators: NDArray[Generator]
    load: float

    def __post_init__(self):
        self.optimize_power = cached(self.optimize_power)

    def optimize_power(self, bitstring: str) -> OptimizeResult:
        """ Finds optimal generation cost for a given generator assignment. """
        def generation_cost_total(powers: list[float]) -> float:
            return sum(gen.generation_cost(power) for gen, power in zip(enabled_generators, powers))

        enabled_generators = self.generators[[int(val) == 1 for val in bitstring]]
        initial_point = [gen.power_range[1] for gen in enabled_generators]
        bounds = [gen.power_range for gen in enabled_generators] if enabled_generators.size > 0 else None
        constraints = [{"type": "ineq", "fun": lambda powers: sum(powers) - self.load}]
        result = optimize.minimize(generation_cost_total, initial_point, method="SLSQP", bounds=bounds, constraints=constraints)
        result.penalty = get_penalty(result.x, constraints)
        result.total = result.fun + result.penalty
        return result

    def evaluate(self, bitstring: str) -> float:
        return self.optimize_power(bitstring).total


class PowerFlowProblem:
    """ Generalized version of GeneratorCommitmentProblem, where locations of generators and loads are taken into account.
    Specifically, the problem is described by a graph, where nodes represent neighborhoods that can include generators and loads.
    Generated power is consumed by local loads. Any excess power can be transferred to adjacent nodes to supplement their generators.
    Edges represent power lines between neighborhoods and have finite capacities, so routing the generated power to the loads now becomes a problem too. """

    def __init__(self, graph: Graph):
        """ Graph nodes should have the following properties:
        1) generators: list[Generator]. List of generator instances located at a given node.
        2) load: float >= 0. Total load at a given node.
        Graph edges should have the following properties:
        1) capacity: float > 0. Maximum power that can be routed through a given edge.
        The following additional properties will be automatically added to the graph:
        1) var_inds: list[int], added to nodes. List of indices in the optimization vector to which generator's power outputs at this node map.
        2) var_ind: int, added to edges. Index in the optimization vector to which power flow through this edge maps.
        3) start: node key type, added to edges. Defines positive flow direction by choosing an arbitrary end of a given edge as start.
        Collects generators from all nodes into a single generators list. """
        self.graph = graph
        self.generators = np.array([gen for _, gens in self.graph.nodes(data="generators") for gen in gens])
        self.optimize_power = cached(self.optimize_power)

        var_ind = 0
        for _, data in self.graph.nodes(data=True):
            data["var_inds"] = list(range(var_ind, var_ind + len(data["generators"])))
            var_ind += len(data["generators"])
        nx.set_edge_attributes(self.graph, {(u, v): {"var_ind": i + var_ind, "start": u} for i, (u, v) in enumerate(self.graph.edges)})

    def evaluate_power_balance(self, powers: list[float], node_label: Hashable) -> float:
        """ Evaluates power balance at a given node, i.e. sum of all generated powers + incoming - outgoing - load. """
        power_balance = 0
        this_node = self.graph.nodes[node_label]
        for gen_ind in this_node["var_inds"]:
            power_balance += powers[gen_ind]
        for _, v, data in self.graph.edges(node_label, data=True):
            power_balance += powers[data["var_ind"]] * (-1) ** (node_label == data["start"])
        power_balance -= this_node["load"]
        return power_balance

    def get_constraints(self) -> list[dict[str, Any]]:
        """ Provides a list of inequality constraints. Each constraint requires power balance at a given node to be non-negative.
        Positive power balance (i.e. excess power) is allowed, since it is assumed that each node has a power sink. """
        constraints = []
        for label in self.graph.nodes:
            constraints.append({"type": "ineq", "fun": partial(self.evaluate_power_balance, node_label=label)})
        return constraints

    def optimize_power(self, bitstring: str) -> OptimizeResult:
        """ Finds optimal power vector for a given set of enabled generators, defined by the bitstring. """
        def generation_cost_total(powers: list[float]) -> float:
            return sum(gen.generation_cost(power) for gen, power in zip(self.generators, powers))

        bounds = [np.array(gen.power_range) * int(bitstring[i]) for i, gen in enumerate(self.generators)]
        bounds += [(-cap, cap) for _, _, cap in self.graph.edges(data="capacity")]
        initial_point = [bound[1] for bound in bounds[:len(self.generators)]]
        initial_point += [0] * len(self.graph.edges)
        constraints = self.get_constraints()
        result = optimize.minimize(generation_cost_total, initial_point, method="SLSQP", bounds=bounds, constraints=constraints)
        result.penalty = get_penalty(result.x, constraints)
        result.total = result.fun + result.penalty
        return result

    def evaluate(self, bitstring: str) -> float:
        """ Returns optimal generation cost + penalty for a given set of enabled generators. """
        return self.optimize_power(bitstring).total
