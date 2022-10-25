import math
from collections import defaultdict

from agents import Evacuee, GuideAgent


class FeatureExtractor:
    area_broadcast_count = defaultdict(lambda: 0)  # Static field, shared among agents
    action_position_map = {'UL': (-1, +1), 'UM': (0, +1), 'UR': (+1, +1), 'ML': (-1, 0), 'MM': (0, 0), 'MR': (+1, 0),
                           'DL': (-1, -1), 'DM': (0, -1), 'DR': (+1, -1)}

    def __init__(self, model, guide_agent):
        self.model = model
        self.guide_agent = guide_agent

    def get_state(self):
        g = self.guide_agent

        # my position
        g_x, g_y = my_pos = g.pos

        # closest sensor
        closest_sensor, closest_sensor_distance = self.get_closest_sensor()

        # new evacuees informed about area
        informed_evacuees = FeatureExtractor.area_broadcast_count[closest_sensor.area_id]
        newly_informed_evacuees = 0

        for e in self.model.grid.get_neighbors(g.pos, self.model.moore):
            if type(e) == Evacuee and e.assigned_exit_area_id is None:
                newly_informed_evacuees += 1

        informed_evacuees += newly_informed_evacuees
        FeatureExtractor.area_broadcast_count[closest_sensor.area_id] = informed_evacuees
        uninformed_evacuees_in_area = closest_sensor.evacuees_in_area - informed_evacuees

        # closest_exit_position
        closest_exit_id, closest_exit_distance = self.get_closest_exit()

        # closest other guide
        area_map = self.model.exits_maps[closest_exit_id]
        closest_guide, closest_guide_distance = self.get_closest_guide(area_map)

        features = {
            # 'my_pos': my_pos,

            # 'closest_sensor_position': closest_sensor.pos,
            'closest_sensor_distance': closest_sensor_distance,
            # 'closest_sensor_evacuees': closest_sensor.evacuees_in_area,
            'newly_informed_evacuees': newly_informed_evacuees,
            'uninformed_evacuees_in_area': uninformed_evacuees_in_area,

            'closest_exit_id': closest_exit_id,
            'closest_exit_distance': closest_exit_distance,

            'closest_guide_distance': closest_guide_distance,
        }

        return features

    def get_next_state(self, action):
        g = self.guide_agent

        # simulate step
        current_count = FeatureExtractor.area_broadcast_count.copy()
        current_pos = g.pos
        next_pos = FeatureExtractor.action_to_position(current_pos, action)

        g.move(next_pos)
        next_state = self.get_state()

        # reverse simulation
        FeatureExtractor.area_broadcast_count = current_count
        g.move(current_pos)

        return next_state

    def get_reward(self, previous_state, next_state):
        if self.model.schedule.get_breed_count(Evacuee) > 0:
            if next_state['newly_informed_evacuees'] > 0:
                return 10
            else:
                return -1

        elif previous_state["closest_exit_distance"] > next_state["closest_exit_distance"]:
            return 10
        else:
            return -1

    @staticmethod
    def action_to_position(pos, action):
        x, y = pos
        x_mod, y_mod = FeatureExtractor.action_position_map[action]
        return x + x_mod, y + y_mod

    def get_closest_sensor(self):
        closest_sensor = None
        closest_sensor_distance = self.model.max_route_len
        for s in self.model.sensors:
            # s_x, s_y = s.pos
            # dst = abs(area_map[g_x][g_y] - area_map[s_x][s_y])
            dst = math.dist(self.guide_agent.pos, s.pos)

            if dst < closest_sensor_distance:
                closest_sensor_distance = dst
                closest_sensor = s

        return closest_sensor, closest_sensor_distance

    def get_closest_exit(self):
        x, y = self.guide_agent.pos

        closest_exit_id = None
        closest_exit_distance = self.model.max_route_len
        for k, v in self.model.exits_maps.items():
            dst = v[x][y]

            if dst < closest_exit_distance:
                closest_exit_distance = dst
                closest_exit_id = k

        return closest_exit_id, closest_exit_distance

    def get_closest_guide(self, area_map):
        g_x, g_y = self.guide_agent.pos

        closest_guide = None
        closest_guide_distance = self.model.max_route_len
        for x, y in self.model.guides_positions:
            dst = abs(area_map[g_x][g_y] - area_map[x][y])

            if dst < closest_guide_distance:
                closest_guide_distance = dst
                for agent in self.model.grid.get_cell_list_contents([(x, y)]):
                    if isinstance(agent, GuideAgent):
                        closest_guide = agent
                        break

        return closest_guide, closest_guide_distance
