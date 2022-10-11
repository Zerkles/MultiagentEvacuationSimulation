from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import Evacuee, Guide, Obstacle, Exit
from schedule import RandomActivationByBreed
import random
from itertools import product


class EvacuationModel(Model):
    verbose = False  # Print-monitoring

    description = (
        "A model for simulating area evacuation. Consists of many evacuees and a few cooperating evacuation guides."
    )

    def __init__(self, map_type, evacuees_num, guides_num, agents_clipping, guides_mode, evacuees_random_position,
                 guides_random_position):

        super().__init__()
        # Set parameters
        self.map_type = map_type
        self.evacuees_num = evacuees_num
        self.guides_num = guides_num
        self.agents_clipping = agents_clipping
        self.guides_mode = guides_mode
        self.evacuees_random_position = evacuees_random_position
        self.guides_random_position = guides_random_position

        self.height = self.width = 100
        self.exits = [(75, 100, 0), (0, 25, 99)]  # (x1,x2,y), x1<x2

        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=True)
        self.datacollector = DataCollector(
            {
                "Evacuees": lambda m: m.schedule.get_breed_count(Evacuee),
                "Guides": lambda m: m.schedule.get_breed_count(Guide),
            }
        )

        available_locations = list(product(range(self.width), range(self.height)))

        # Place exits
        for area_num, exit in enumerate(self.exits):
            x1, x2, y = exit

            for xi in range(x1, x2):
                exit = Exit(uid=self.next_id(), pos=(xi, y), model=self, area_num=area_num)
                self.grid.place_agent(exit, exit.pos)
                self.schedule.add(exit)

                available_locations.remove((xi, y))

        # Place obstacles

        # Place guides
        if self.guides_random_position:
            for _ in range(self.guides_num):
                pos = random.choice(available_locations)
                available_locations.remove(pos)

                guide = Guide(uid=self.next_id(), pos=pos, model=self)
                self.grid.place_agent(guide, pos)
                self.schedule.add(guide)
        else:
            pass

        # Place evacuees
        if self.evacuees_random_position:
            for _ in range(self.evacuees_num):
                pos = random.choice(available_locations)
                available_locations.remove(pos)

                evacuee = Evacuee(uid=self.next_id(), pos=pos, model=self)
                self.grid.place_agent(evacuee, pos)
                self.schedule.add(evacuee)
        else:
            pass

        self.running = True
        self.datacollector.collect(self)


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

    def run_model(self, step_count=200):
        if self.verbose:
            print("Initial number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Initial number of Guides: ", self.schedule.get_breed_count(Guide))

        for i in range(step_count):
            self.step()

        if self.verbose:
            print("")
            print("Final number of Evacuees: ", self.schedule.get_breed_count(Evacuee))
            print("Final number of Guides: ", self.schedule.get_breed_count(Guide))
