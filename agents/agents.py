from random import Random
from typing import Tuple, Set, Dict

from simulation.simulation_state import SimulationState


class StateAgent:
    """Base class for a model agent."""

    def __init__(self, unique_id: int, pos: Tuple[int, int], random_seed: Random) -> None:
        self.unique_id = unique_id
        self.pos = pos
        self.random_seed = random_seed

    def step(self, state: SimulationState) -> None:
        pass

    @property
    def random(self) -> Random:
        return self.random_seed

    def on_remove(self, state) -> None:
        pass


class Obstacle(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: Random) -> None:
        super().__init__(uid, pos, random_seed)


class Exit(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: Random, exit_area_id: int) -> None:
        super().__init__(uid, pos, random_seed)

        self.area_id = exit_area_id


class Sensor(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: Random, sensor_area_id: int,
                 sensing_positions: Set[Tuple[int, int]]) -> None:
        super().__init__(uid, pos, random_seed)

        self.sensor_area_id = sensor_area_id
        self.sensing_positions = sensing_positions
        self.evacuees_in_area = None

    def step(self, state: SimulationState) -> None:
        self.evacuees_in_area = len(self.sensing_positions.intersection(state.grid.positions_by_breed[Evacuee]))


class MapInfo(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], value: int, color: str, random_seed: Random) -> None:
        super().__init__(uid, pos, random_seed)

        self.value = value
        self.color = color


class GuideAgent(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: Random) -> None:
        super().__init__(uid, pos, random_seed)

    def get_closest_exit(self, state: SimulationState) -> Tuple[int, int]:
        x, y = state.schedule.agents_by_breed[type(self)][self.unique_id].pos

        closest_exit_id = None
        closest_exit_distance = state.max_route_len
        for k, v in state.exit_maps.items():
            dst = v[x][y]

            if dst < closest_exit_distance:
                closest_exit_distance = dst
                closest_exit_id = k

        return closest_exit_id, closest_exit_distance

    def get_exit(self, state: SimulationState) -> None:
        pass


class Evacuee(StateAgent):

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: Random) -> None:
        super().__init__(uid, pos, random_seed)
        self.assigned_exit_area_id = None

    def step(self, state: SimulationState) -> str:
        if self.assigned_exit_area_id is None:
            return "MM"

        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents

        legal_actions_with_positions = state.grid.get_legal_actions_with_positions(pos, ghost_agents)

        if legal_actions_with_positions == {}:
            return "MM"

        legal_actions_with_distances = self.get_distance_for_positions(state, legal_actions_with_positions)

        best_action = min(legal_actions_with_distances, key=legal_actions_with_distances.get)

        return best_action

    def get_distance_for_positions(self, state: SimulationState,
                                   legal_actions_with_positions: Dict[str, Tuple[int, int]]) -> Dict[str, int]:
        """Returns dictionary {action:distance}"""

        area_map = state.exit_maps[self.assigned_exit_area_id]
        legal_actions_with_distances = dict()

        for k, v in legal_actions_with_positions.items():
            x, y = v
            legal_actions_with_distances.update({k: area_map[x][y]})

        return legal_actions_with_distances
