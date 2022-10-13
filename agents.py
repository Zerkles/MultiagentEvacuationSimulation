import math
from collections import Counter
from statistics import mean

from mesa import Agent


class WalkingAgent(Agent):
    # Point (0x,0y) is in LEFT BOTTOM; U-Up, D-Down, L-Left, M-Middle, R-Right;
    directions = ["DL", "ML", "UL", "DM", "MM", "UM", "DR", "MR", "UR"]

    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model
        self.moore = True

    def get_legal_actions(self):
        legal_positions = dict()
        x, y = self.pos
        neighbourhood = self.model.area_positions_from_points((x - 1, y - 1), (x + 1, y + 1))

        for i, pos in enumerate(neighbourhood):

            if self.model.grid.out_of_bounds(pos):
                continue

            agents_at_pos = [type(x) for x in self.model.grid.get_cell_list_contents([pos])]
            if Obstacle in agents_at_pos:
                continue

            if not self.model.ghost_agents and (Evacuee in agents_at_pos or Guide in agents_at_pos):
                continue

            legal_positions.update({WalkingAgent.directions[i]: pos})

        if legal_positions == dict():
            legal_positions = {"MM": self.pos}

        return legal_positions

    def move(self, pos):
        self.model.grid.move_agent(self, pos)

    def on_escape(self):
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

    def broadcast_exit_id(self, exit_id):
        neighbors = self.model.grid.get_cell_list_contents(self.model.grid.get_neighborhood(self.pos, True))
        for n in neighbors:
            if type(n) == Evacuee:
                n.assigned_exit_area_id = exit_id


class Evacuee(WalkingAgent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, pos, model)
        self.assigned_exit_area_id = None

    def step(self):
        if self.assigned_exit_area_id is None:
            return

        if self.model.evacuees_share_information:
            self.broadcast_exit_id(self.assigned_exit_area_id)

        legal_positions = list(self.get_legal_actions().values())
        legal_positions_goals = self.calculate_delta_goal(legal_positions)

        best_position = min(legal_positions_goals, key=legal_positions_goals.get)
        self.move(best_position)

    def calculate_delta_goal(self, legal_positions):
        legal_positions_distances = dict()
        for pos in legal_positions:
            distances = []

            for exit_pos in self.model.exit_areas_positions[self.assigned_exit_area_id]:
                distances.append(math.dist(pos, exit_pos))

            legal_positions_distances.update({pos: mean(distances)})  # min/mean ? mean is much faster

        return legal_positions_distances


class Guide(WalkingAgent):
    alpha = 0.5
    beta = 0.1
    theta = 3

    def __init__(self, uid, pos, model, mode):
        super().__init__(uid, pos, model)
        self.mode = mode
        self.direction_change_timer = 0
        self.closest_exit_id = 1

        if mode == "A":
            self.step = self.step_mode_a
        else:
            self.step = self.step_mode_b

    def step_mode_a(self):
        # Escape broadcast
        closest_exit = None

        for k, v in self.model.exit_areas_positions.items():
            distances = []
            for exit_pos in v:
                distances.append(math.dist(self.pos, exit_pos))
            mean_distance = mean(distances)

            if closest_exit is None:
                closest_exit = (k, mean_distance)
            elif closest_exit[1] > mean_distance:
                closest_exit = (k, mean_distance)

        self.closest_exit_id = closest_exit[0]

        self.broadcast_exit_id(self.closest_exit_id)

        # Theta timer
        if self.direction_change_timer > 0:
            self.direction_change_timer -= 1
            return

        # Moving
        legal_positions = list(self.get_legal_actions().values())
        legal_positions_goals = self.calculate_delta_goal(legal_positions)

        best_position = min(legal_positions_goals, key=legal_positions_goals.get)
        self.move(best_position)

    def step_mode_b(self):
        self.get_legal_actions()

    def calculate_delta_goal(self, legal_positions):
        legal_positions_with_goal = {}
        for pos in legal_positions:
            for s in self.model.sensors:
                guides = self.model.guides.copy()
                guides.remove(self)

                direction_pull = s.evacuees_in_area / (math.dist(pos, s.pos) + 0.0001)  # to avoid zero division
                direction_repulsion = Guide.beta / mean([math.dist(g.pos, pos) for g in guides])
                goal = Guide.alpha * direction_pull + (1 - Guide.alpha) * direction_repulsion

            legal_positions_with_goal.update({pos: goal})

        return legal_positions_with_goal


class Obstacle(Agent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos


class Exit(Agent):
    def __init__(self, uid, pos, model, exit_area_id):
        super().__init__(uid, model)
        self.area_num = exit_area_id
        self.pos = pos

    def step(self):
        agents = self.model.grid.get_cell_list_contents(self.pos)

        for agent in agents:
            if type(agent) is Evacuee or type(agent) is Guide:
                agent.on_escape()


class Sensor(Agent):
    def __init__(self, uid, pos, model, area_id, sensing_area):
        super().__init__(uid, model)
        self.pos = pos
        self.area_id = area_id
        self.sensing_positions = sensing_area
        self.evacuees_in_area = None

    def step(self):
        agents = [type(x) for x in self.model.grid.get_cell_list_contents(self.sensing_positions)]
        self.evacuees_in_area = Counter(agents).get(Evacuee)
