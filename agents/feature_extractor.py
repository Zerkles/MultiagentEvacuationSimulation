import math
from collections import defaultdict

import numpy as np

from agents.agents import GuideAgent, Evacuee, Sensor
from simulation.grid import EvacuationGrid
from simulation.simulation_state import SimulationState


class FeatureExtractor:
    unvisited_positions = set(EvacuationGrid.area_positions_from_points((0, 0), (99, 99)))
    informed_evacuees = 0

    def __init__(self, guide_id):
        self.guide_id = guide_id

    def get_guide_obj(self, state: SimulationState):
        return state.schedule.get_agent_by_id(self.guide_id)

    def get_features(self, state: SimulationState, action=None):
        # my next position
        guide = self.get_guide_obj(state)
        pos = guide.pos
        if action is not None:
            pos = state.grid.action_to_position(pos, action)
        else:
            pos = pos

        # closest sensor
        closest_sensor, closest_sensor_distance = FeatureExtractor.get_closest_sensor(state, pos, normalize=True)

        # informed evacuees
        newly_informed_evacuees, uninformed_evacuees = FeatureExtractor.get_newly_informed_evacuees(state, pos,
                                                                                                    normalize=True)

        # closest_exit_position
        closest_exit_id, closest_exit_distance = FeatureExtractor.get_closest_exit(state, pos, normalize=True)

        # closest other guide
        area_map, _ = state.grid.generate_square_rounded_map(pos, {pos})
        max_area_route_len = np.amax(area_map)
        closest_guide, closest_guide_distance = self.get_closest_guide(state, pos, area_map, max_area_route_len,
                                                                       normalize=True)

        # visited positions
        closest_unvisited_position = FeatureExtractor.get_closest_unvisited_position(state, pos, area_map,
                                                                                     max_area_route_len,
                                                                                     normalize=True)

        features = {
            'bias': 1.0,

            'closest_sensor_distance': closest_sensor_distance,
            'newly_informed_evacuees': newly_informed_evacuees,
            'uninformed_evacuees': uninformed_evacuees,

            # 'closest_exit_id': closest_exit_id,
            'closest_exit_distance': closest_exit_distance,

            'closest_guide_distance': closest_guide_distance,

            'closest_unvisited_position': closest_unvisited_position,
        }

        return features

    def get_reward(self, state, action, next_state):
        next_feats = self.get_features(state, action)
        feats = self.get_features(state)

        # print("f", feats)
        # print("nf", next_feats)

        if len(next_state.schedule.agents_by_breed[Evacuee]) > 0:
            if feats['newly_informed_evacuees'] > 0:
                return 10.0
            else:
                return -1.0

        elif feats["closest_exit_distance"] > next_feats["closest_exit_distance"]:
            return 10.0
        else:
            return -1.0

    @staticmethod
    def get_newly_informed_evacuees(state: SimulationState, pos, normalize=True, update_variables=False):
        newly_informed_evacuees = 0

        for n in state.grid.get_neighbors(pos, True, include_center=False):
            if type(n) == Evacuee and n.assigned_exit_area_id is None:
                newly_informed_evacuees += 1

        uninformed_evacuees = state.evacuees_num - (FeatureExtractor.informed_evacuees + newly_informed_evacuees)

        if update_variables:
            FeatureExtractor.informed_evacuees += newly_informed_evacuees

        if normalize:
            newly_informed_evacuees = FeatureExtractor.normalize(newly_informed_evacuees, 9)
            uninformed_evacuees = FeatureExtractor.normalize(uninformed_evacuees, state.evacuees_num)

        return newly_informed_evacuees, uninformed_evacuees

    @staticmethod
    def get_closest_sensor(state: SimulationState, pos, normalize=True):
        closest_sensor = None
        closest_sensor_distance = state.max_route_len
        for uid, agent in state.schedule.agents_by_breed[Sensor].items():
            dst = math.dist(pos, agent.pos)

            if dst < closest_sensor_distance:
                closest_sensor_distance = dst
                closest_sensor = agent

        if normalize:
            closest_sensor_distance = FeatureExtractor.normalize(closest_sensor_distance, state.width * math.sqrt(2))

        return closest_sensor, closest_sensor_distance

    @staticmethod
    def get_closest_exit(state: SimulationState, pos, normalize=True):
        x, y = pos

        closest_exit_id = None
        closest_exit_distance = state.max_route_len
        for k, v in state.exit_maps.items():
            dst = v[x][y]

            if dst < closest_exit_distance:
                closest_exit_distance = dst
                closest_exit_id = k

        if normalize:
            closest_exit_distance = FeatureExtractor.normalize(closest_exit_distance,
                                                               np.amax(state.exit_maps[closest_exit_id]))

        return closest_exit_id, closest_exit_distance

    def get_closest_guide(self, state, pos, area_map, max_area_route_len, normalize=True):
        g_x, g_y = pos
        guide_map_val = area_map[g_x][g_y]

        closest_guide = None
        closest_guide_distance = max_area_route_len
        for breed in state.schedule.agents_by_breed.keys():
            if issubclass(breed, GuideAgent):
                for uid, agent in state.schedule.agents_by_breed[breed].items():
                    if uid == self.guide_id:
                        continue

                    x, y = agent.pos
                    dst = abs(guide_map_val - area_map[x][y])

                    if dst <= closest_guide_distance:
                        closest_guide_distance = dst
                        closest_guide = agent

        if normalize:
            closest_guide_distance = FeatureExtractor.normalize(closest_guide_distance, max_area_route_len)

        return closest_guide, closest_guide_distance

    @staticmethod
    def get_closest_unvisited_position(state, pos, area_map, max_area_route_len, normalize=True):
        g_x, g_y = pos
        guide_map_val = area_map[g_x][g_y]

        unvisited_positions = FeatureExtractor.unvisited_positions - set(
            state.grid.get_neighborhood(pos, True, include_center=True))

        closest_guide_distance = max_area_route_len

        for x, y in unvisited_positions:
            dist = abs(guide_map_val - area_map[x][y])

            if dist < closest_guide_distance:
                closest_guide_distance = dist

        if normalize:
            closest_guide_distance = FeatureExtractor.normalize(closest_guide_distance, max_area_route_len)

        return closest_guide_distance

    def update_extractor(self, state: SimulationState, action):
        guide = self.get_guide_obj(state)
        next_pos = state.grid.action_to_position(guide.pos, action)

        # informed evacuees
        self.get_newly_informed_evacuees(state, next_pos, update_variables=True)

        # visited positions
        FeatureExtractor.unvisited_positions -= set(state.grid.get_neighborhood(next_pos, True, include_center=True))

    @staticmethod
    def normalize(value, max_value):
        return value / max_value
