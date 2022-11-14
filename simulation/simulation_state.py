from copy import copy


class SimulationState:

    def __init__(self, grid, schedule, exit_maps, args):
        self.grid = grid
        self.schedule = schedule
        self.exit_maps = exit_maps

        self.__dict__.update(args)
