from mesa import Agent


class WalkingAgent(Agent):

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

        legal_positions.add(self.pos)

        return legal_positions

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

            if type(self) == GuideAgent or n.assigned_exit_area_id is None:
                n.assigned_exit_area_id = exit_id


class GuideAgent(WalkingAgent):
    # Point (0x,0y) is in LEFT BOTTOM; U-Up, D-Down, L-Left, M-Middle, R-Right;
    directions = ["DL", "ML", "UL", "DM", "MM", "UM", "DR", "MR", "UR"]

    def __init__(self, uid, pos, model, positions_set):
        super().__init__(uid, pos, model, positions_set)

    def get_closest_exit(self):
        x, y = self.pos

        closest_exit_id = None
        closest_exit_distance = self.model.max_route_len
        for k, v in self.model.exits_maps.items():
            dst = v[x][y]

            if dst < closest_exit_distance:
                closest_exit_distance = dst
                closest_exit_id = k

        return closest_exit_id, closest_exit_distance

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


class Evacuee(WalkingAgent):
    def __init__(self, uid, pos, model, positions_set):
        super().__init__(uid, pos, model, positions_set)
        self.assigned_exit_area_id = None

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
        map = self.model.exits_maps[self.assigned_exit_area_id]
        legal_positions_distances = dict()

        for pos in legal_positions:
            legal_positions_distances.update({pos: map[pos[0]][pos[1]]})

        return legal_positions_distances


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
            if type(agent) is Evacuee or isinstance(agent, GuideAgent):
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
    def __init__(self, uid, pos, model, value, color):
        super().__init__(uid, model)
        self.pos = pos
        self.value = value
        self.color = color
