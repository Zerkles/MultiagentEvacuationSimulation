from mesa import Agent


class Evacuee(Agent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model
        moore = True

    def step(self):
        pass


class Guide(Agent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model
        moore = True

    def step(self):
        pass


class Obstacle(Agent):
    def __init__(self, uid, pos, model):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model


class Exit(Agent):
    def __init__(self, uid, pos, model, area_num):
        super().__init__(uid, model)
        self.pos = pos
        self.model = model
        self.area_num = area_num
