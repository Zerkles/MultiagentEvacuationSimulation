from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Tuple, Set, Dict

import numpy as np
from mesa.space import MultiGrid

from agents.agents import Obstacle, Evacuee, GuideAgent


class EvacuationGrid(MultiGrid):
    # Point (0x,0y) is in LEFT BOTTOM; U-Up, D-Down, L-Left, M-Middle, R-Right;
    action_position_map = {'UL': (-1, +1), 'UM': (0, +1), 'UR': (+1, +1), 'ML': (-1, 0), 'MM': (0, 0), 'MR': (+1, 0),
                           'DL': (-1, -1), 'DM': (0, -1), 'DR': (+1, -1)}

    def __init__(self, width: int, height: int, torus: bool) -> None:
        super().__init__(width, height, torus)
        self.positions_by_breed = defaultdict(lambda: set())

    # def __deepcopy__(self, memodict={}):
    #     new_instance = EvacuationGrid(self.width, self.height, self.torus)
    #     new_instance.positions_by_breed = deepcopy(self.positions_by_breed)
    #     new_instance.grid = self.grid
    #     return new_instance

    def get_legal_positions(self, pos, ghost_agents) -> Set[Tuple[int, int]]:
        legal_positions = set(self.get_neighborhood(pos, True, include_center=False, radius=1))

        legal_positions -= self.positions_by_breed[Obstacle]
        if not ghost_agents:
            legal_positions -= (self.positions_by_breed[Evacuee].union(self.positions_by_breed[GuideAgent]))

        # legal_positions.add(pos)
        return legal_positions

    def get_legal_actions_with_positions(self, pos, ghost_agents) -> Dict[str, Tuple[int, int]]:
        x, y = pos
        all_actions = dict()
        for k, (x_mod, y_mod) in EvacuationGrid.action_position_map.items():
            all_actions.update({k: (x + x_mod, y + y_mod)})

        legal_positions = self.get_legal_positions(pos, ghost_agents)
        legal_actions = dict()

        for k, v in all_actions.items():
            if v in legal_positions:
                legal_actions.update({k: v})

        return legal_actions

    def get_legal_actions(self, pos, ghost_agents):
        return list(self.get_legal_actions_with_positions(pos, ghost_agents).keys())


    @staticmethod
    def area_positions_from_points(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        xs = sorted([x1, x2])
        ys = sorted([y1, y2])

        return list(product(range(xs[0], xs[1] + 1), range(ys[0], ys[1] + 1)))

    def action_to_position(self, pos, action):
        x, y = pos
        x_mod, y_mod = self.action_position_map[action]
        return x + x_mod, y + y_mod

    def generate_square_rounded_map(self, start_position, exit_positions):
        area_map = np.empty((self.width, self.height), int)
        unmeasured_positions = set(
            EvacuationGrid.area_positions_from_points((0, 0), (self.width - 1, self.height - 1))) - \
                               self.positions_by_breed[Obstacle]

        current_positions = {start_position}
        distance = 0

        unreachable_positions = set()
        av_pos_len = len(unmeasured_positions)

        while unmeasured_positions != set():
            for x, y in current_positions:
                area_map[x][y] = distance

            next_positions = set()

            if distance % 2 == 0:
                for pos in current_positions:
                    next_positions.update(self.get_neighborhood(pos, moore=True))
            else:
                for pos in current_positions:
                    next_positions.update(self.get_neighborhood(pos, moore=False))

            unmeasured_positions -= current_positions
            current_positions = next_positions.intersection(unmeasured_positions)

            distance += 1

            # anti-stuck, helps in situation when some positions are unreachable
            if av_pos_len == len(unmeasured_positions):
                unreachable_positions = unmeasured_positions
                break
            else:
                av_pos_len = len(unmeasured_positions)

        for x, y in unreachable_positions:
            area_map[x][y] = np.amax(area_map)

        for x, y in exit_positions:
            area_map[x][y] = 0

        for x, y in self.positions_by_breed[Obstacle]:
            area_map[x][y] = np.amax(area_map)

        return area_map, unreachable_positions
