from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Package
import random
import numpy as np


def get_nearest_package(env: WarehouseEnv, robot_id: int) -> Package:
    robot, packages = env.get_robot(robot_id), env.packages[0:2]
    distances = [manhattan_distance(robot.position, package.position) for package in packages if package.on_board]
    return packages[np.argmin(distances)]

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot_id = 0 if robot_id == 1 else 1
    opponent = env.get_robot(other_robot_id)
    cant_get_credit = False

    h = 4.5*robot.battery + 4*robot.credit
    if robot.package is None:
        nearest_package = get_nearest_package(env, robot_id)
        package_delivery_dist = manhattan_distance(nearest_package.position, nearest_package.position)
        nearest_package_dist = manhattan_distance(robot.position, nearest_package.position)
        h += package_delivery_dist - nearest_package_dist
        cant_get_credit = robot.battery <= nearest_package_dist
    else:
        credit_from_delivery = 2*manhattan_distance(robot.package.position, robot.package.destination)
        steps_to_delivery = manhattan_distance(robot.position, robot.package.destination)
        h += credit_from_delivery - steps_to_delivery
        cant_get_credit = robot.battery <= steps_to_delivery

    if cant_get_credit and opponent.package:
        h += 1000 * (10 - manhattan_distance(robot.position, opponent.package.destination))

    return h





class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        children_heuristics = [self.heuristic(child, robot_id) for child in children]
        for heuristic in children_heuristics:
            print(f"heuristic: {heuristic}")
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)