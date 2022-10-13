from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import Evacuee, Guide, Obstacle, Exit, Sensor
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
        # Set parameters
        self.width = width
        self.height = height
        self.map_type = map_type
        self.evacuees_num = evacuees_num
        self.guides_num = guides_num
        self.ghost_agents = ghost_agents
        self.guides_mode = guides_mode
        self.evacuees_share_information = evacuees_share_information
        self.guides_random_position = guides_random_position

        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=True)
        self.datacollector = DataCollector(
            {
                "Evacuees": lambda m: m.schedule.get_breed_count(Evacuee),
            }
        )

        # CONFIG
        available_positions = self.area_positions_from_points((0, 0), (99, 99))
        areas_centers = [(25, 75), (75, 25), (25, 25), (75, 75)]

        # Place exits
        exits_points = [((0, 0), (25, 0)), ((75, 99), (99, 99))]
        self.exit_areas_positions = dict()

        for area_id, exit_obj in enumerate(exits_points):
            area = self.area_positions_from_points(exit_obj[0], exit_obj[1])
            self.exit_areas_positions.update({area_id: []})

            for pos in area:
                exit_obj = Exit(uid=self.next_id(), pos=pos, model=self, exit_area_id=area_id)
                self.grid.place_agent(exit_obj, pos)
                self.schedule.add(exit_obj)

                self.exit_areas_positions[area_id].append(pos)
                available_positions.remove(pos)

        # Place obstacles
        obstacles_points = []
        if self.map_type == 'cross':
            obstacles_points = [((49, 9), (50, 39)), ((49, 89), (50, 59)), ((9, 49), (39, 50)), ((59, 49), (89, 50))]

        elif self.map_type == 'boxes':
            thickness = 10
            obstacles_points = [((pos[0] + thickness, pos[1] + thickness), (pos[0] - thickness, pos[1] - thickness)) for
                                pos in areas_centers]

        obstacles_positions = []
        for a, b in obstacles_points:
            obstacles_positions.extend(self.area_positions_from_points(a, b))

        for pos in obstacles_positions:
            available_positions.remove(pos)
            obstacle = Obstacle(uid=self.next_id(), pos=pos, model=self)
            self.grid.place_agent(obstacle, pos)
            self.schedule.add(obstacle)

        # Place sensors
        self.sensors = []
        for i, pos in enumerate(areas_centers):
            sensing_area = self.area_positions_from_points((pos[0] - 25, pos[1] - 25), (pos[0] + 25, pos[1] + 25))
            sensing_area = set(available_positions).intersection(set(sensing_area))
            sensor = Sensor(uid=self.next_id(), pos=pos, model=self, area_id=i, sensing_area=sensing_area)
            self.grid.place_agent(sensor, pos)
            self.schedule.add(sensor)
            self.sensors.append(sensor)

        # Place guides
        self.guides = []
        for i in range(self.guides_num):
            if self.guides_random_position or self.map_type == 'boxes':
                pos = random.choice(available_positions)
            else:
                pos = areas_centers[i]

            available_positions.remove(pos)
            guide = Guide(uid=self.next_id(), pos=pos, model=self, mode=self.guides_mode)
            self.grid.place_agent(guide, pos)
            self.schedule.add(guide)
            self.guides.append(guide)

        # Place evacuees
        for _ in range(self.evacuees_num):
            pos = random.choice(available_positions)
            available_positions.remove(pos)

            evacuee = Evacuee(uid=self.next_id(), pos=pos, model=self)
            self.grid.place_agent(evacuee, pos)
            self.schedule.add(evacuee)

        self.running = True
        self.datacollector.collect(self)

    # TODO: This method should be used by server, but it is not for some reason
    #  (then there is "stop if" in step function)
    def run_model(self):
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
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print(
                [
                    self.schedule.time,
                    self.schedule.get_breed_count(Evacuee),
                    self.schedule.get_breed_count(Guide),
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
