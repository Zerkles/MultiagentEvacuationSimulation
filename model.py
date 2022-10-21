import itertools
import math
import sys
from collections import Counter
from math import sqrt
from statistics import mean, median

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import Evacuee, Guide, Obstacle, Exit, Sensor, MapInfo
from schedule import RandomActivationByBreed
import random
from itertools import product


class EvacuationModel(Model):
    verbose = False

    description = (
        "A model for simulating area evacuation. Consists of many evacuees and a few cooperating evacuation guides."
    )

    def __init__(self, width, height, guides_mode, map_type, evacuees_num, guides_num, ghost_agents,
                 evacuees_share_information, guides_random_position, show_map, rectangles_num, rectangles_max_size,
                 erosion_proba):

        super().__init__()
        # Mapping parameters
        self.width = width
        self.height = height
        self.map_type = map_type
        self.ghost_agents = ghost_agents
        self.guides_mode = guides_mode
        self.evacuees_share_information = evacuees_share_information

        random_rectangles_params = {"rectangles_num": rectangles_num, "rectangles_max_size": rectangles_max_size,
                                    "erosion_proba": erosion_proba}

        # CONFIG
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.datacollector = DataCollector(
            {
                "Evacuees": lambda m: m.schedule.get_breed_count(Evacuee),
            }
        )

        self.moore = True
        self.max_route_len = (self.width * self.height) + 1

        fixed_positions = {'x_1_4': int(self.width / 4), 'x_1_2': int(self.width / 2),
                           'x_3_4': int(self.width - (self.width / 4)),
                           'y_1_4': int(self.height / 4), 'y_1_2': int(self.height / 2),
                           'y_3_4': int(self.height - (self.height / 4))}

        areas_centers = [(fixed_positions['x_1_4'], fixed_positions['y_3_4']),
                         (fixed_positions['x_3_4'], fixed_positions['y_1_4']),
                         (fixed_positions['x_1_4'], fixed_positions['y_1_4']),
                         (fixed_positions['x_3_4'], fixed_positions['y_3_4'])]

        available_positions = self.area_positions_from_points((0, 0), (self.width - 1, self.height - 1))

        exits_areas_corners = [((0, 0), (25, 0)), ((75, 99), (99, 99))]
        self.exits_positions = self.init_exits(available_positions, exits_areas_corners)

        self.obstacles_positions = self.init_obstacles(available_positions, areas_centers, fixed_positions,
                                                       random_rectangles_params)

        self.exits_maps, unreachable_positions = self.init_exits_maps(self.exits_positions, show_map=show_map)

        available_positions = list(set(available_positions) - unreachable_positions)

        self.sensors = self.init_sensors(available_positions, areas_centers, fixed_positions)
        self.guides_positions = self.init_guides(guides_num, guides_random_position, available_positions, areas_centers)
        self.evacuees_positions = self.init_evacuees(evacuees_num, available_positions)

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
        for breed in [Exit, Sensor, Guide]:
            self.schedule.step_breed(breed)

        dct = {k: self.exits_maps[v.assigned_exit_area_id][v.pos[0]][v.pos[1]] for k, v in
               self.schedule.agents_by_breed[Evacuee].items()}
        order = sorted(dct, key=dct.get)
        self.schedule.step_breed_ordered(Evacuee, order)

        # collect data
        self.datacollector.collect(self)

        if self.schedule.get_breed_count(Evacuee) == 0:
            self.running = False

        if self.verbose:
            print([self.schedule.time, self.schedule.get_breed_count(Evacuee)])

    def init_obstacles(self, available_positions, areas_centers, fixed_positions, random_rectangles_params):
        obstacles_corners = []
        obstacles_positions = set()

        if self.map_type == 'default':
            return set()
        elif self.map_type == 'cross':
            gap_thck = 10  # gap thickness
            obstacles_corners = [((fixed_positions['x_1_2'], 0 + gap_thck),
                                  (fixed_positions['x_1_2'], fixed_positions['y_1_2'] - gap_thck)),

                                 ((fixed_positions['x_1_2'], self.height - 1 - gap_thck),
                                  (fixed_positions['x_1_2'], fixed_positions['y_1_2'] + gap_thck)),

                                 ((0 + gap_thck, fixed_positions['y_1_2']),
                                  (fixed_positions['x_1_2'] - gap_thck, fixed_positions['y_1_2'])),

                                 ((fixed_positions['x_1_2'] + gap_thck, fixed_positions['y_1_2']),
                                  (self.height - gap_thck, fixed_positions['y_1_2']))]

        elif self.map_type == 'boxes':
            thck = 15  # thickness
            for x, y in areas_centers:
                obstacles_corners.append(((x + thck, y + thck), (x - thck, y - thck)))

        elif self.map_type == 'random_rectangles':
            rectangles_num = random_rectangles_params['rectangles_num']
            rectangle_max_size = range(1, random_rectangles_params['rectangles_max_size'])
            erosion_proba = random_rectangles_params['erosion_proba']

            obstacles_positions = set()
            for _ in range(rectangles_num):
                center = random.choice(available_positions)

                x_random = random.choice(rectangle_max_size)
                max_x, min_x = center[0] + x_random, center[0] - x_random

                y_random = random.choice(rectangle_max_size)
                max_y, min_y = center[1] + y_random, center[1] - y_random

                box_points = set(self.area_positions_from_points((min_x, min_y), (max_x, max_y)))
                rectangle_points = set()
                for x, y in box_points:
                    if x == max_x or x == min_x or y == max_y or y == min_y:
                        rectangle_points.add((x, y))

                rectangle_points = rectangle_points.intersection(set(available_positions))

                for pos in list(rectangle_points):
                    if random.choices([True, False], weights=[erosion_proba, 1 - erosion_proba], k=1)[0]:
                        rectangle_points.remove(pos)

                obstacles_positions.update(rectangle_points)

        if self.map_type == 'cross' or self.map_type == 'boxes':
            for i, (a, b) in enumerate(obstacles_corners):
                area_positions = self.area_positions_from_points(a, b)
                obstacles_positions.update(area_positions)

        for pos in obstacles_positions:
            available_positions.remove(pos)

            obstacle = Obstacle(uid=self.next_id(), pos=pos, model=self)
            self.grid.place_agent(obstacle, pos)

        return obstacles_positions

    def init_exits(self, available_positions, exits_areas_corners):
        exits_positions = dict()
        for area_id, exit_obj in enumerate(exits_areas_corners):
            area = self.area_positions_from_points(exit_obj[0], exit_obj[1])
            exits_positions.update({area_id: area})

            for pos in area:
                exit_obj = Exit(uid=self.next_id(), pos=pos, model=self, exit_area_id=area_id)
                self.grid.place_agent(exit_obj, pos)
                self.schedule.add(exit_obj)

                available_positions.remove(pos)

        return exits_positions

    def init_exits_maps(self, exits_positions, show_map=False):
        exits_maps = dict()
        unreachable_positions = set()
        for k, v in exits_positions.items():
            start_position = (int(median([x[0] for x in v])), int(median(([x[1] for x in v]))))

            area_map, unreachable_positions_part = self.generate_square_rounded_map(start_position, v)

            exits_maps[k] = area_map
            unreachable_positions.update(unreachable_positions_part)

        # Map test
        if show_map:
            test_map = list(exits_maps.values())[1]
            for x in range(100):
                for y in range(100):
                    map_obj = MapInfo(uid=self.next_id(), pos=(x, y), model=self,
                                      value=test_map[x][y], color="transparent")
                    self.grid.place_agent(map_obj, (x, y))

        return exits_maps, unreachable_positions

    def init_sensors(self, available_positions, areas_centers, fixed_positions):
        sensors = set()
        for i, pos in enumerate(areas_centers):
            sensing_area = self.area_positions_from_points(
                (pos[0] - fixed_positions['x_1_4'], pos[1] - fixed_positions['y_1_4']),
                (pos[0] + fixed_positions['x_1_4'], pos[1] + fixed_positions['y_1_4']))
            sensing_area = set(available_positions).intersection(set(sensing_area))

            sensor = Sensor(uid=self.next_id(), pos=pos, model=self, area_id=i, sensing_positions=sensing_area)
            self.grid.place_agent(sensor, pos)
            self.schedule.add(sensor)

            sensors.add(sensor)

        return sensors

    def init_guides(self, guides_num, guides_random_position, available_positions, areas_centers):
        guides_positions = set()
        for i in range(guides_num):
            if guides_random_position or self.map_type == 'boxes' or self.map_type == 'random_rectangles':
                pos = random.choice(available_positions)
            else:
                pos = areas_centers[i]

            guides_positions.add(pos)
            available_positions.remove(pos)

            guide = Guide(uid=self.next_id(), pos=pos, model=self, mode=self.guides_mode,
                          positions_set=guides_positions)
            self.grid.place_agent(guide, pos)
            self.schedule.add(guide)

        return guides_positions

    def init_evacuees(self, evacuees_num, available_positions):
        if evacuees_num > len(available_positions):
            evacuees_num = len(available_positions)

        evacuees_positions = set()
        for _ in range(evacuees_num):
            pos = random.choice(available_positions)
            evacuees_positions.add(pos)
            available_positions.remove(pos)

            evacuee = Evacuee(uid=self.next_id(), pos=pos, model=self, positions_set=evacuees_positions)
            self.grid.place_agent(evacuee, pos)
            self.schedule.add(evacuee)

        return evacuees_positions

    @staticmethod
    def area_positions_from_points(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        xs = sorted([x1, x2])
        ys = sorted([y1, y2])

        return list(product(range(xs[0], xs[1] + 1), range(ys[0], ys[1] + 1)))

    def generate_square_rounded_map(self, start_position, exit_positions):
        area_map = np.empty((self.width, self.height), int)
        unmeasured_positions = set(
            self.area_positions_from_points((0, 0), (self.width - 1, self.height - 1))) - self.obstacles_positions

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
                    next_positions.update(self.grid.get_neighborhood(pos, moore=True))
            else:
                for pos in current_positions:
                    next_positions.update(self.grid.get_neighborhood(pos, moore=False))

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
            area_map[x][y] = int(self.max_route_len)

        for x, y in exit_positions:
            area_map[x][y] = 0

        for x, y in self.obstacles_positions:
            area_map[x][y] = int(self.max_route_len)

        return area_map, unreachable_positions
