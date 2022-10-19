import itertools
import math
import sys
from collections import Counter
from math import sqrt
from statistics import mean

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import Evacuee, Guide, Obstacle, Exit, Sensor, MapInfo
from schedule import RandomActivationByBreed
import random
from itertools import product


class EvacuationModel(Model):
    verbose = False  # Print-monitoring

    description = (
        "A model for simulating area evacuation. Consists of many evacuees and a few cooperating evacuation guides."
    )

    def __init__(self, width, height, guides_mode, map_type, evacuees_num, guides_num, ghost_agents,
                 evacuees_share_information, guides_random_position):

        super().__init__()
        # Mapping parameters
        self.width = width
        self.height = height
        self.map_type = map_type
        self.evacuees_num = evacuees_num
        self.guides_num = guides_num
        self.ghost_agents = ghost_agents
        self.guides_mode = guides_mode
        self.evacuees_share_information = evacuees_share_information
        self.guides_random_position = guides_random_position

        # CONFIG
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.datacollector = DataCollector(
            {
                "Evacuees": lambda m: m.schedule.get_breed_count(Evacuee),
            }
        )

        self.moore = True
        self.diagonal = sqrt(self.width ** 2 + self.height ** 2)
        available_positions = self.area_positions_from_points((0, 0), (99, 99))
        areas_centers = [(24, 74), (74, 24), (24, 24), (74, 74)]

        # OBSTACLES
        obstacles_points = []
        if self.map_type == 'cross':
            obstacles_points = [((49, 9), (50, 39)), ((49, 89), (50, 59)), ((9, 49), (39, 50)), ((59, 49), (89, 50))]

        elif self.map_type == 'boxes':
            thk = 10  # thickness
            for x, y in areas_centers:
                obstacles_points.append(((x + thk, y + thk), (x - thk, y - thk)))

        obstacles_dict = dict()
        self.obstacles_positions = set()
        for i, (a, b) in enumerate(obstacles_points):
            area_positions = self.area_positions_from_points(a, b)
            self.obstacles_positions.update(area_positions)
            obstacles_dict[i] = self.area_positions_from_points(a, b)

        for pos in self.obstacles_positions:
            available_positions.remove(pos)

            obstacle = Obstacle(uid=self.next_id(), pos=pos, model=self)
            self.grid.place_agent(obstacle, pos)

        # EXITS
        exits_points = [((0, 0), (25, 0)), ((75, 99), (99, 99))]

        self.exit_areas_positions = dict()
        for area_id, exit_obj in enumerate(exits_points):
            area = self.area_positions_from_points(exit_obj[0], exit_obj[1])
            self.exit_areas_positions.update({area_id: area})

            for pos in area:
                exit_obj = Exit(uid=self.next_id(), pos=pos, model=self, exit_area_id=area_id)
                self.grid.place_agent(exit_obj, pos)
                self.schedule.add(exit_obj)

                available_positions.remove(pos)

        self.exit_areas_maps = dict()
        for k, v in self.exit_areas_positions.items():
            start_position = (int(mean([x[0] for x in v])), int(mean(([x[1] for x in v]))))

            # area_map = self.generate_round_map(start_position, v)
            area_map = self.generate_square_map(start_position, v)

            # area_map = self.generate_obstacles_border(area_map, start_position, obstacles_points)
            # area_map = self.generate_raytracing_map(start_position, v)

            self.exit_areas_maps[k] = area_map

        # Map test
        test_map = list(self.exit_areas_maps.values())[1]
        for x in range(100):
            for y in range(100):
                map_obj = MapInfo(uid=self.next_id(), pos=(x, y), model=self, value=test_map[x][y], color="transparent")
                self.grid.place_agent(map_obj, (x, y))

        # Place SENSORS
        self.sensors = []
        for i, pos in enumerate(areas_centers):
            sensing_area = self.area_positions_from_points((pos[0] - 25, pos[1] - 25), (pos[0] + 25, pos[1] + 25))
            sensing_area = set(available_positions).intersection(set(sensing_area))

            sensor = Sensor(uid=self.next_id(), pos=pos, model=self, area_id=i, sensing_positions=sensing_area)
            self.grid.place_agent(sensor, pos)
            self.schedule.add(sensor)

            self.sensors.append(sensor)

        # GUIDES
        self.guides_positions = set()
        for i in range(self.guides_num):
            if self.guides_random_position or self.map_type == 'boxes':
                pos = random.choice(available_positions)
            else:
                pos = areas_centers[i]

            self.guides_positions.add(pos)
            available_positions.remove(pos)

            guide = Guide(uid=self.next_id(), pos=pos, model=self, mode=self.guides_mode,
                          positions_set=self.guides_positions)
            self.grid.place_agent(guide, pos)
            self.schedule.add(guide)

        # EVACUEES
        self.evacuees_positions = set()
        for _ in range(self.evacuees_num):
            pos = random.choice(available_positions)
            self.evacuees_positions.add(pos)
            available_positions.remove(pos)

            evacuee = Evacuee(uid=self.next_id(), pos=pos, model=self, positions_set=self.evacuees_positions)
            self.grid.place_agent(evacuee, pos)
            self.schedule.add(evacuee)

        self.running = True
        self.datacollector.collect(self)

    def run_model(self):
        # TODO: This method should be used by server, but it is not for some reason
        #  (then there is "stop if" in step function)

        if self.verbose:
            print("Initial number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Initial number of Guides: ", self.schedule.get_breed_count(Guide))

        while self.schedule.get_breed_count(Evacuee) > 0:
            self.step()

        if self.verbose:
            print("")
            print("Final number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Final number of Guides: ", self.schedule.get_breed_count(Guide))

    def step(self):
        for breed in [Exit, Sensor, Guide, Evacuee]:
            self.schedule.step_breed(breed)

        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print(
                [
                    self.schedule.time,
                    self.schedule.get_breed_count(Evacuee),
                ]
            )

        if self.schedule.get_breed_count(Evacuee) == 0:
            self.running = False

    @staticmethod
    def area_positions_from_points(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        xs = sorted([x1, x2])
        ys = sorted([y1, y2])

        return list(product(range(xs[0], xs[1] + 1), range(ys[0], ys[1] + 1)))

    def generate_square_map(self, start_position, exit_positions):

        area_map = np.empty((self.width, self.height), int)
        available_positions = set(self.area_positions_from_points((0, 0), (99, 99))) - self.obstacles_positions

        current_positions = {start_position}

        distance = 0

        while available_positions != set():
            for x, y in current_positions:
                area_map[x][y] = distance

            next_positions = set()
            for pos in current_positions:
                next_positions.update(self.grid.get_neighborhood(pos, moore=False,include_center=True))
                # self.grid.get_neighborhood()

            available_positions -= current_positions
            current_positions = next_positions.intersection(available_positions)

            distance += 1

        for x, y in exit_positions:
            area_map[x][y] = 0

        for x, y in self.obstacles_positions:
            area_map[x][y] = 99

        return area_map

    def generate_round_map(self, start_position, exit_positions):
        area_map = np.empty((self.width, self.height), int)

        for x in range(self.width):
            for y in range(self.height):
                area_map[x][y] = int(math.dist((x, y), start_position))

        for x, y in exit_positions:
            area_map[x][y] = 0

        return area_map

    def generate_raytracing_map(self, start_position, exit_positions):
        area_map = np.full((self.width, self.height), -1)

        area_map[start_position[0]][start_position[1]] = 0

        for x in range(self.width):
            for y in range(self.height):
                area_map = self.shortest_path(area_map, (x, y), start_position)

        # distances_corners = [math.dist(x, exit_pos) for x in positions]
        # best_corner = obstacles_corners[distances_corners.index(min(distances_corners))]
        #
        # return self.shortest_path(map, start_pos, best_corner, obstacles_corners)

        obstacles_corners = []

        counts = dict()
        for pos in self.obstacles_positions:
            counts.update({pos: len(self.grid.get_neighbors(pos, self.moore))})

        min_val = min(counts.values())

        for k, v in list(counts.items()):
            if counts[k] == min_val:
                obstacles_corners.append(k)

        for pos_cor in obstacles_corners:
            x, y = pos_cor
            values_in_neighborhood = dict()

            for x_n, y_n in self.grid.get_neighborhood(pos_cor, self.moore):
                values_in_neighborhood.update({pos_cor: area_map[x_n][y_n]})

            if not all([val == -1 for val in values_in_neighborhood.values()]):
                obstacles_corners.remove(pos_cor)
            elif area_map[x][y] == -1:
                _, val = max(values_in_neighborhood.items(), key=lambda item: item[1])
                area_map[x][y] = val + 1

        for x, y in obstacles_corners:
            info = MapInfo(uid=self.next_id(), pos=(x, y), model=self, value=area_map[x][y], color="red")
            self.grid.place_agent(info, (x, y))

        for x, y in exit_positions:
            area_map[x][y] = 0

        return area_map

    def generate_obstacles_border(self, area_map, area_start_position, obstacle_points):

        for a, b in obstacle_points:
            area_positions = set(self.area_positions_from_points(a, b))

            border_positions = set()
            for pos in area_positions:
                border_positions.update(self.grid.get_neighborhood(pos, moore=self.moore))

            border_positions -= area_positions

            border_positions_with_distance = dict()
            for pos in border_positions:
                border_positions_with_distance.update({pos: math.dist(area_start_position, pos)})

            border_positions_with_distance = dict(
                sorted(border_positions_with_distance.items(), key=lambda item: item[1]))

            min_value = min(border_positions_with_distance.values())

            for enum, k in enumerate(border_positions_with_distance):
                border_positions_with_distance[k] = min_value + 1

            for (x, y), val in border_positions_with_distance.items():
                area_map[x][y] = val

        return area_map

    def shortest_path(self, map, start_pos, exit_pos):
        route = [start_pos]
        x1, y1 = pos = start_pos

        if start_pos in self.obstacles_positions:
            return map

        while map[x1][y1] == -1:
            positions = self.grid.get_neighborhood(pos, moore=self.moore)
            distances = [math.dist(x, exit_pos) for x in positions]
            best_position = positions[distances.index(min(distances))]

            if best_position in self.obstacles_positions:
                return map

            route.append(best_position)
            x1, y1 = pos = best_position

        x_last, y_last = route[-1]
        distance = map[x_last][y_last]

        if distance == -1:
            return map

        for x, y in reversed(route):
            map[x][y] = distance
            distance += 1

        return map
