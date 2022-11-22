import math
import sys
from collections import defaultdict

import multiprocess
import numpy as np
from multiprocess.context import Process

from agents.agents import GuideAgent, Evacuee, Sensor
from simulation.grid import EvacuationGrid
from simulation.simulation_state import SimulationState


def get_distance_maps(positions_chunk, grid, queue):
    chunk_maps = dict()
    chunk_maps_lists = dict()
    for pos in positions_chunk:
        area_map, _ = grid.generate_square_rounded_map(pos, {pos})
        chunk_maps[pos] = area_map

        # TODO: Version List (position = [positions sorted by distance value])

        # lst = []
        # for x in range(area_map.shape[0]):
        #     for y in range(area_map.shape[1]):
        #         lst.append((x, y, area_map[x][y]))
        #
        # # lst.remove(pos)
        # lst.sort(key=lambda i: i[2])
        # # chunk_maps_lists[pos] = [(x[0], x[1]) for x in lst]
        # chunk_maps_lists[pos] = lst

        # TODO: Version Dict({position:{distance_value:[list_of_positions]}})

        # chunk_maps_lists[pos] = dict()
        # for val in np.unique(area_map):
        #     results = np.where(area_map == val)
        #     chunk_maps_lists[pos][val] = set(zip(results[0], results[1]))

        # print(list(zip(results[0],results[1])), v)
        # print(area_map)
        # input()
    print("Process finished")

    # queue.put({'msg': 'lists', 'content': chunk_maps_lists})
    queue.put({'msg': 'maps', 'content': chunk_maps})


def get_feature_extractor_maps(grid, n_jobs=-1):
    width = grid.width
    height = grid.height

    if n_jobs == -1:
        n_threads = 16

    all_positions = EvacuationGrid.area_positions_from_points((0, 0), (width - 1, height - 1))

    chunks = []
    step = math.ceil(len(all_positions) / n_threads)
    for i in range(0, len(all_positions), step):
        chunks.append(all_positions[i:i + step])

    global get_distance_maps

    queue = multiprocess.Queue(maxsize=100)
    # processes = []
    for chunk in chunks:
        p = Process(target=get_distance_maps, args=(chunk, grid, queue,)).start()
        # processes.append(p)

    maps = dict()
    maps_lists = dict()

    for _ in range(n_threads):
        item = queue.get()

        if item['msg'] == 'lists':
            maps_lists.update(item['content'])
        else:
            maps.update(item['content'])
    # for p in processes:
    #     p.join()

    return maps, maps_lists


class FeatureExtractor:
    informed_evacuees = 0
    maps = dict()
    maps_lists = dict()
    unvisited_positions = set()

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
        area_map = FeatureExtractor.maps[pos]
        max_area_route_len = np.amax(area_map)

        closest_guide, closest_guide_distance = self.get_closest_guide(state, pos, area_map, max_area_route_len,
                                                                       normalize=True)

        # visited positions
        closest_unvisited_position = FeatureExtractor.get_closest_unvisited_position(state, pos, max_area_route_len,
                                                                                     normalize=True)

        features = {
            'bias': 1.0,

            # 'closest_sensor_distance': closest_sensor_distance,

            'newly_informed_evacuees': newly_informed_evacuees,
            'uninformed_evacuees': uninformed_evacuees,

            'closest_exit_distance': closest_exit_distance,

            'closest_guide_distance': closest_guide_distance,

            'closest_unvisited_position': closest_unvisited_position,
        }

        # features.update(FeatureExtractor.get_all_sensor_distance_and_evacuees(state, pos))

        # divide all by 10 to avoid gradient explosion
        features = {k: v / 10 for k, v in features.items()}

        return features

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
            closest_sensor_distance = FeatureExtractor.normalize(closest_sensor_distance,
                                                                 (state.width - 1) * math.sqrt(2))

        return closest_sensor, closest_sensor_distance

    @staticmethod
    def get_all_sensor_distance_and_evacuees(state: SimulationState, pos, normalize=True):
        # closest_sensor = None
        # closest_sensor_distance = state.max_route_len

        # TODO: Połączyć to w jakiś jeden faktor?

        dct = dict()
        for uid, sensor in state.schedule.agents_by_breed[Sensor].items():
            dst = math.dist(pos, sensor.pos)
            dct.update({f"sensor_{uid}_dst": dst, f"sensor_{uid}_ev": sensor.evacuees_in_area})

        if normalize:
            for k, v in dct.items():
                if "dst" in k:
                    dct[k] = FeatureExtractor.normalize(v, (state.width - 1) * math.sqrt(2))
                else:
                    dct[k] = FeatureExtractor.normalize(v, len(
                        state.schedule.get_agent_by_id(int(k.split("_")[1])).sensing_positions))

        return dct

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
    def get_closest_unvisited_position(state, pos, max_area_route_len, normalize=True):
        g_x, g_y = pos
        area_map = FeatureExtractor.maps[pos]
        guide_map_val = area_map[g_x][g_y]

        # unvisited_positions = FeatureExtractor.unvisited_positions - set(
        #     state.grid.get_neighborhood(pos, True, include_center=True))

        unvisited_positions = FeatureExtractor.unvisited_positions

        closest_unvisited_position_distance = max_area_route_len

        for x, y in unvisited_positions:
            dist = abs(guide_map_val - area_map[x][y])

            if dist < closest_unvisited_position_distance:
                closest_unvisited_position_distance = dist

        if normalize:
            closest_unvisited_position_distance = FeatureExtractor.normalize(closest_unvisited_position_distance,
                                                                             max_area_route_len)
            closest_unvisited_position_distance -= 1

        return closest_unvisited_position_distance

    def update_extractor(self, feats, next_state: SimulationState):

        # informed evacuees
        FeatureExtractor.informed_evacuees += int(feats['newly_informed_evacuees'] * 9)

        # visited positions
        guide = self.get_guide_obj(next_state)

        x, y = next_state.grid.action_to_position(guide.pos, guide.last_action)
        last_pos = (x * -1, y * -1)
        visited_positions = next_state.grid.get_neighborhood(last_pos, True, include_center=True)

        FeatureExtractor.unvisited_positions -= set(visited_positions)
        # for pos in visited_positions:
        #     FeatureExtractor.maps_lists[last_pos].remove(pos)

    @staticmethod
    def normalize(value, max_value):
        return value / max_value
