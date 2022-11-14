import json

import os

import time
from collections import defaultdict
from copy import deepcopy
from statistics import median
from typing import Dict

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from agents.agents import Obstacle, Exit, Sensor, MapInfo, StateAgent, GuideAgent, Evacuee
from agents.agents_guides import GuideABT, GuideQLearning
import random

from simulation.grid import EvacuationGrid
from simulation.schedule import EvacuationScheduler
from simulation.simulation_state import SimulationState


class EvacuationModel(Model):
    verbose = False

    description = (
        "A model for simulating area evacuation. Consists of many evacuees and a few cooperating evacuation guides."
    )

    def __init__(self, width, height, guides_mode, map_type, evacuees_num, guides_num, ghost_agents,
                 evacuees_share_information, guides_random_position, show_map, rectangles_num, rectangles_max_size,
                 erosion_proba, cross_gap, boxes_thickness, qlearning_params):

        super().__init__()

        # Mapping parameters
        self.width = width
        self.height = height
        self.map_type = map_type
        self.ghost_agents = ghost_agents
        self.guides_mode = guides_mode
        self.evacuees_share_information = evacuees_share_information

        # copy for sim_state purposes
        self.evacuees_num = evacuees_num

        map_params = {"rectangles_num": rectangles_num, "rectangles_max_size": rectangles_max_size,
                      "erosion_proba": erosion_proba, "cross_gap": cross_gap,
                      "boxes_thickness": boxes_thickness}

        # CONFIG
        self.schedule = EvacuationScheduler(self)
        self.grid = EvacuationGrid(self.height, self.width, torus=False)
        self.datacollector = DataCollector(
            {
                "Evacuees": lambda m: m.schedule.get_breed_count(Evacuee),
            }
        )

        self.moore = True
        self.max_route_len = (self.width * self.height) + 1
        self.qlearning_params = None

        fixed_positions = {'x_1_4': int(self.width / 4), 'x_1_2': int(self.width / 2),
                           'x_3_4': int(self.width - (self.width / 4)),
                           'y_1_4': int(self.height / 4), 'y_1_2': int(self.height / 2),
                           'y_3_4': int(self.height - (self.height / 4))}

        areas_centers = [(fixed_positions['x_1_4'], fixed_positions['y_3_4']),
                         (fixed_positions['x_3_4'], fixed_positions['y_1_4']),
                         (fixed_positions['x_1_4'], fixed_positions['y_1_4']),
                         (fixed_positions['x_3_4'], fixed_positions['y_3_4'])]

        available_positions = EvacuationGrid.area_positions_from_points((0, 0), (self.width - 1, self.height - 1))

        # EXITS
        exits_areas_corners = [((0, 0), (25, 0)), ((75, 99), (99, 99))]
        exits_positions = self.init_exits(available_positions, exits_areas_corners)
        self.grid.positions_by_breed[Exit] = exits_positions

        # OBSTACLES
        obstacles_positions = self.init_obstacles(available_positions, areas_centers, fixed_positions,
                                                  map_params)
        self.grid.positions_by_breed[Obstacle] = obstacles_positions

        # EXITS MAPS
        exits_maps, unreachable_positions = self.init_exits_maps(exits_positions, show_map=show_map)
        self.exit_maps = exits_maps

        available_positions = list(set(available_positions) - unreachable_positions)

        # SENSORS
        sensors_positions = self.init_sensors(available_positions, areas_centers, fixed_positions)
        self.grid.positions_by_breed[Sensor] = sensors_positions

        # GUIDES
        guides_positions = self.init_guides(guides_num, guides_random_position, available_positions, areas_centers,
                                            qlearning_params)
        self.grid.positions_by_breed.update(guides_positions)

        # EVACUEES
        evacuees_positions = self.init_evacuees(evacuees_num, available_positions)
        self.grid.positions_by_breed[Evacuee] = evacuees_positions

        self.datacollector.collect(self)

    def run_model(self):
        # This method is not invoked by server!

        simulation_start_time = time.time()

        if self.verbose:
            print("Initial number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Initial number of Guides: ", self.schedule.get_breed_count(GuideAgent))

        while self.running:
            self.step()

        if self.verbose:
            print("")
            print("Final number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Final number of Guides: ", self.schedule.get_breed_count(GuideAgent))

        return time.time() - simulation_start_time

    def step(self):
        self.schedule.step()

        # Terminal conditions
        for guide in self.schedule.get_breed_agents(GuideQLearning):
            if guide.score <= 0:
                self.remove_agent(guide, self.get_simulation_state())

        if self.schedule.get_breed_count(Evacuee) == 0 and self.schedule.get_guides_count() == 0:
            self.running = False
            self.on_remove(self.get_simulation_state())

        elif self.schedule.get_breed_count(Evacuee) != 0 and self.schedule.get_guides_count() == 0:
            self.running = False
            self.on_remove(self.get_simulation_state())

        # Collect data
        self.datacollector.collect(self)

        if self.verbose:
            print([self.schedule.time, self.schedule.get_breed_count(Evacuee)])

    def on_remove(self, state):
        print(self.schedule.steps)
        for agent in self.schedule.get_breed_agents(GuideQLearning):
            self.remove_agent(agent, state)

    def move_agent(self, agent: StateAgent, action: str):
        pos = self.grid.action_to_position(agent.pos, action)

        self.grid.positions_by_breed[type(agent)].discard(agent.pos)
        self.grid.positions_by_breed[type(agent)].add(pos)

        self.grid.move_agent(agent, pos)

    def remove_agent(self, agent: StateAgent, state):
        self.grid.positions_by_breed[type(agent)].remove(agent.pos)

        if type(agent) == GuideQLearning:
            agent.on_remove(state)
            self.add_guide_experience(vars(agent))

        self.grid.remove_agent(agent)
        self.schedule.remove(agent)

    def broadcast_exit_info(self, agent: StateAgent, exit_id: int, force: bool = False):
        neighbors = self.grid.get_neighbors(agent.pos, self.moore)

        neighbor_evacuees = set(self.schedule.get_breed_agents(Evacuee)).intersection(neighbors)

        for n in neighbor_evacuees:

            if n.assigned_exit_area_id is not None and not force:
                continue
            else:
                n.assigned_exit_area_id = exit_id

    def add_guide_experience(self, guide_vars: Dict):
        if self.qlearning_params is None:
            self.qlearning_params = guide_vars
        # else:
        #     for k, v in vars:
        #         self.qlearning_params[k] = (self.qlearning_params[k] + vars[k]) / 2

    def init_obstacles(self, available_positions, areas_centers, fixed_positions, map_params):
        obstacles_corners = []
        obstacles_positions = set()

        if self.map_type == 'default':
            pass
        elif self.map_type == 'cross':
            gap_thck = map_params["cross_gap"]  # gap thickness
            obstacles_corners = [((fixed_positions['x_1_2'], 0 + gap_thck),
                                  (fixed_positions['x_1_2'], fixed_positions['y_1_2'] - gap_thck)),

                                 ((fixed_positions['x_1_2'], self.height - 1 - gap_thck),
                                  (fixed_positions['x_1_2'], fixed_positions['y_1_2'] + gap_thck)),

                                 ((0 + gap_thck, fixed_positions['y_1_2']),
                                  (fixed_positions['x_1_2'] - gap_thck, fixed_positions['y_1_2'])),

                                 ((fixed_positions['x_1_2'] + gap_thck, fixed_positions['y_1_2']),
                                  (self.height - gap_thck, fixed_positions['y_1_2']))]

        elif self.map_type == 'boxes':
            thck = map_params["boxes_thickness"]  # thickness
            for x, y in areas_centers:
                obstacles_corners.append(((x + thck, y + thck), (x - thck, y - thck)))

        elif self.map_type == 'random_rectangles':
            rectangles_num = map_params['rectangles_num']
            rectangle_max_size = range(1, map_params['rectangles_max_size'])
            erosion_proba = map_params['erosion_proba']

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
            obstacle = Obstacle(uid=self.next_id(), pos=pos, random_seed=self.random)
            self.grid.place_agent(obstacle, pos)
            self.schedule.add(obstacle)

            available_positions.remove(pos)

        return obstacles_positions

    def init_exits(self, available_positions, exits_areas_corners):
        exits_positions = dict()
        for area_id, exit_obj in enumerate(exits_areas_corners):
            area = EvacuationGrid.area_positions_from_points(exit_obj[0], exit_obj[1])
            exits_positions.update({area_id: area})

            for pos in area:
                exit_obj = Exit(uid=self.next_id(), pos=pos, random_seed=self.random, exit_area_id=area_id)
                self.grid.place_agent(exit_obj, pos)
                self.schedule.add(exit_obj)

                available_positions.remove(pos)

        return exits_positions

    def init_exits_maps(self, exits_positions, show_map=False):
        exits_maps = dict()
        unreachable_positions = set()
        for k, v in exits_positions.items():
            start_position = (int(median([x[0] for x in v])), int(median(([x[1] for x in v]))))

            area_map, unreachable_positions_part = self.grid.generate_square_rounded_map(start_position, v)

            exits_maps[k] = area_map
            unreachable_positions.update(unreachable_positions_part)

        # Map test
        if show_map:
            test_map = list(exits_maps.values())[1]
            for x in range(100):
                for y in range(100):
                    map_obj = MapInfo(uid=self.next_id(), pos=(x, y), random_seed=self.random,
                                      value=test_map[x][y], color="transparent")
                    self.grid.place_agent(map_obj, (x, y))

        return exits_maps, unreachable_positions

    def init_sensors(self, available_positions, areas_centers, fixed_positions):
        sensors_positions = set()
        for i, pos in enumerate(areas_centers):
            sensing_area = EvacuationGrid.area_positions_from_points(
                (pos[0] - fixed_positions['x_1_4'], pos[1] - fixed_positions['y_1_4']),
                (pos[0] + fixed_positions['x_1_4'], pos[1] + fixed_positions['y_1_4']))
            sensing_area = set(available_positions).intersection(set(sensing_area))

            sensor = Sensor(uid=self.next_id(), pos=pos, random_seed=self.random, sensor_area_id=i,
                            sensing_positions=sensing_area)
            self.grid.place_agent(sensor, pos)
            self.schedule.add(sensor)

            sensors_positions.add(pos)

        return sensors_positions

    def init_guides(self, guides_num, guides_random_position, available_positions, areas_centers, q_learning_params):
        guides_positions = defaultdict(lambda: set())
        for i in range(guides_num):
            if guides_random_position or self.map_type == 'boxes' or self.map_type == 'random_rectangles':
                pos = random.choice(available_positions)
            else:
                pos = areas_centers[i]

            if self.guides_mode == "Q Learning":
                qlearning_weights = defaultdict(lambda: 0.0)
                if q_learning_params['weights'] is not None:
                    qlearning_weights = q_learning_params['weights']

                elif os.path.exists("output/weights.txt"):
                    with open("output/weights.txt", "r") as f:
                        qlearning_weights.update(dict(json.load(f)))

                epsilon = q_learning_params['epsilon']
                gamma = q_learning_params['gamma']  # aka discount factor
                alpha = q_learning_params['alpha']

                guide = GuideQLearning(uid=self.next_id(), pos=pos, random_seed=self.random, epsilon=epsilon,
                                       gamma=gamma, alpha=alpha, weights=qlearning_weights)

            else:
                guide = GuideABT(uid=self.next_id(), pos=pos, random_seed=self.random, mode=self.guides_mode,
                                 args=vars(self))

            self.grid.place_agent(guide, pos)
            self.schedule.add(guide)

            available_positions.remove(pos)
            guides_positions[type(guide)].add(pos)
        return guides_positions

    def init_evacuees(self, evacuees_num, available_positions):
        if evacuees_num > len(available_positions):
            evacuees_num = len(available_positions)

        evacuees_positions = set()
        for _ in range(evacuees_num):
            pos = random.choice(available_positions)

            evacuee = Evacuee(uid=self.next_id(), pos=pos, random_seed=self.random)
            self.grid.place_agent(evacuee, pos)
            self.schedule.add(evacuee)

            evacuees_positions.add(pos)
            available_positions.remove(pos)

        return evacuees_positions



    def get_simulation_state(self, deep=False):
        params_keys = ['width', 'height', 'guides_mode', 'map_type', 'evacuees_num', 'ghost_agents',
                       'evacuees_share_information', 'max_route_len']
        params = {k: v for k, v in vars(self).items() if k in params_keys}
        exit_maps = self.exit_maps

        if deep:
            grid = deepcopy(self.grid)
            schedule = deepcopy(self.schedule)
        else:
            grid = self.grid
            schedule = deepcopy(self.schedule)

        return SimulationState(grid, schedule, exit_maps, params)
