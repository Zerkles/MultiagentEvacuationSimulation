import math
from collections import Counter
import random
from statistics import mean

from mesa import Agent


class WalkingAgent(Agent):
    # Point (0x,0y) is in LEFT BOTTOM; U-Up, D-Down, L-Left, M-Middle, R-Right;
    directions = ["DL", "ML", "UL", "DM", "MM", "UM", "DR", "MR", "UR"]

    def __init__(self, uid, pos, model, positions_set):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model
        self.moore = True
        self.positions_set = positions_set

    def get_legal_positions(self):
        legal_positions = set(self.model.grid.get_neighborhood(self.pos, self.moore, include_center=False, radius=1))

        legal_positions -= self.model.obstacles_positions
        if not self.model.ghost_agents:
            legal_positions -= (self.model.evacuees_positions.union(self.model.guides_positions))


        # if not legal_positions:
        #     legal_positions.add(self.pos)
        legal_positions.add(self.pos)

        return legal_positions

    def get_legal_actions(self):

        x, y = self.pos
        generated_neighbourhood = self.model.area_positions_from_points((x - 1, y - 1), (x + 1, y + 1))

        all_actions = dict(zip(generated_neighbourhood, self.directions))
        legal_positions = self.get_legal_positions()

        legal_actions = dict()

        for k, v in all_actions.items():
            if k in legal_positions:
                legal_actions.update({k: v})

        return {value: key for (key, value) in legal_actions.items()}

    def move(self, pos):
        self.positions_set.discard(self.pos)
        self.positions_set.add(pos)
        self.model.grid.move_agent(self, pos)

    def remove(self):
        self.positions_set.discard(self.pos)
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

    def broadcast_exit_id(self, exit_id):
        neighbors = self.model.grid.get_neighbors(self.pos, self.moore)

        for n in neighbors:
            if type(n) != Evacuee:
                continue

            if type(self) == Guide or n.assigned_exit_area_id is None:
                n.assigned_exit_area_id = exit_id


class Evacuee(WalkingAgent):
    def __init__(self, uid, pos, model, positions_set):
        super().__init__(uid, pos, model, positions_set)
        # self.assigned_exit_area_id = random.choice([0, 1])
        self.assigned_exit_area_id = 1

    def step(self):
        if self.assigned_exit_area_id is None:
            return

        elif self.model.evacuees_share_information:
            self.broadcast_exit_id(self.assigned_exit_area_id)

        legal_positions = self.get_legal_positions()
        legal_positions_distances = self.get_distance_for_positions(legal_positions)

        best_position = min(legal_positions_distances, key=legal_positions_distances.get)
        self.move(best_position)

    def get_distance_for_positions(self, legal_positions):
        map = self.model.exit_areas_maps[self.assigned_exit_area_id]
        legal_positions_distances = dict()

        for pos in legal_positions:
            legal_positions_distances.update({pos: map[pos[0]][pos[1]]})

        return legal_positions_distances


class Guide(WalkingAgent):
    alpha = 0.5
    beta = 0.1
    theta = 3

    def __init__(self, uid, pos, model, mode, positions_set):
        super().__init__(uid, pos, model, positions_set)
        self.mode = mode

        if mode == "A":
            self.step = self.step_mode_a
            self.direction_change_timer = 0
            self.closest_exit_id = 1
        elif mode == "B":
            self.step = self.step_mode_b

    def step_mode_a(self):
        # Escape broadcast
        self.closest_exit_id = self.get_closest_exit()
        self.broadcast_exit_id(self.closest_exit_id)

        # Theta timer
        if self.direction_change_timer > 0:
            self.direction_change_timer -= 1
            return

        # Moving
        legal_positions = self.get_legal_positions()
        print(legal_positions)
        legal_positions_goals = self.calculate_delta_goal(legal_positions)

        best_position = min(legal_positions_goals, key=legal_positions_goals.get)
        self.move(best_position)

    def step_mode_b(self):
        self.get_legal_actions()

    def calculate_delta_goal(self, legal_positions):
        guides_positions = self.model.guides_positions.copy()
        guides_positions.remove(self.pos)

        if guides_positions == set():
            return

        legal_positions_with_goal = {}
        for pos in legal_positions:
            for s in self.model.sensors:
                direction_pull = s.evacuees_in_area / (math.dist(pos, s.pos) + 0.0001)  # to avoid zero division

                direction_repulsion = 0
                if guides_positions != set():
                    direction_repulsion = Guide.beta / mean([math.dist(g_pos, pos) for g_pos in guides_positions])

                goal = Guide.alpha * direction_pull + (1 - Guide.alpha) * direction_repulsion

            legal_positions_with_goal.update({pos: goal})

        return legal_positions_with_goal

    def get_closest_exit(self):
        x, y = self.pos

        closest_exit, map = self.model.exit_areas_maps.keys()[1]
        min_dst = self.model.diagonal + 1

        for exit_area_id, map in self.model.exit_areas_maps.items()[1:]:
            dst = map[x][y]

            if dst < min_dst:
                min_dst = dst
                closest_exit = exit_area_id

        return closest_exit


class Obstacle(Agent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos


class Exit(Agent):
    def __init__(self, uid, pos, model, exit_area_id):
        super().__init__(uid, model)
        self.pos = pos
        self.area_id = exit_area_id

    def step(self):
        agents = self.model.grid.get_cell_list_contents(self.pos)

        for agent in agents:
            if type(agent) is Evacuee or type(agent) is Guide:
                agent.remove()


class Sensor(Agent):
    def __init__(self, uid, pos, model, area_id, sensing_positions):
        super().__init__(uid, model)
        self.pos = pos
        self.area_id = area_id
        self.sensing_positions = sensing_positions
        self.evacuees_in_area = None

    def step(self):
        self.evacuees_in_area = len(self.sensing_positions.intersection(self.model.evacuees_positions))

class MapInfo(Agent):
    def __init__(self, uid, pos, model, value,color):
        super().__init__(uid, model)
        self.pos = pos
        self.value = value
        self.color=color

    def step(self):
        pass
