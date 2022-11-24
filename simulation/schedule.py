from collections import defaultdict
from copy import deepcopy
from typing import List, Type

from mesa.time import BaseScheduler

from agents.agents import StateAgent, GuideAgent, Exit, Sensor, Evacuee
from agents.agents_guides import GuideQLearning
from simulation.simulation_state import SimulationState


class EvacuationScheduler(BaseScheduler):

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_breed = defaultdict(dict)

    def add(self, agent: StateAgent) -> None:

        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_breed[agent_class][agent.unique_id] = agent

    def remove(self, agent: StateAgent) -> None:

        del self._agents[agent.unique_id]
        agent_class = type(agent)

        del self.agents_by_breed[agent_class][agent.unique_id]

    def step(self, by_breed: bool = True) -> None:
        if not by_breed:
            super().step()
        else:
            self.steps += 1
            self.time += 1

        state_ref = self.model.get_simulation_state(deep=False)
        # Activate Exits
        for exit_obj in self.get_breed_agents(Exit):
            agents_at_exit = self.model.grid.get_cell_list_contents(exit_obj.pos)
            agents_at_exit.remove(exit_obj)

            for agent in agents_at_exit:
                self.model.remove_agent(agent, state_ref)

        state_ref = self.model.get_simulation_state(deep=False)
        # Activate Sensors
        for sensor in self.get_breed_agents(Sensor):
            sensor.step(state_ref)

        # Activate Evacuees by distance order
        # Determine order
        index_order = dict()
        for evacuee in self.get_breed_agents(Evacuee):
            uid = evacuee.unique_id
            exit_id = evacuee.assigned_exit_area_id
            x, y = evacuee.pos

            if exit_id is not None:
                index_order.update({uid: self.model.exit_maps[exit_id][x][y]})

        order_id = sorted(index_order, key=index_order.get)

        # Activate agents
        for uid in order_id:
            evacuee = self.agents_by_breed[Evacuee][uid]
            action = evacuee.step(state_ref)
            self.model.move_agent(evacuee, action)
            if self.model.evacuees_share_information:
                self.model.broadcast_exit_info(evacuee, evacuee.assigned_exit_area_id)

        state_ref = self.model.get_simulation_state(deep=False)
        # Activate Guides
        for guide in self.get_breed_agents(GuideQLearning):
            # previous_deep_state = self.model.get_simulation_state(deep=True)

            action = guide.step(state_ref)

            feats = guide.extractor.get_features(state_ref, action)
            feats_next = guide.extractor.get_features(state_ref, action)

            self.model.move_agent(guide, action)
            self.model.broadcast_exit_info(guide, guide.get_exit(state_ref), True)

            state_ref = self.model.get_simulation_state()

            reward = guide.get_reward(feats, feats_next)
            guide.update(feats, state_ref, reward)

    def step_breed(self, breed: Type, state: SimulationState, index_order: List[int] = None) -> None:
        if index_order is None:
            agent_keys = list(self.agents_by_breed[breed].keys())
            self.model.random.shuffle(agent_keys)
        else:
            agent_keys = index_order

        for agent_key in agent_keys:
            agent = self.agents_by_breed[breed][agent_key]
            move = agent.step(state)
            self.model.move_agent(agent, move)

    def get_breed_count(self, breed: Type) -> int:
        return len(self.agents_by_breed[breed].values())

    def get_guides_count(self) -> int:
        count = 0
        for k, v in self.agents_by_breed.items():
            if issubclass(k, GuideAgent):
                count += len(v)
        return count

    def get_breed_agents(self, breed: Type) -> List[StateAgent]:
        return list(self.agents_by_breed[breed].values())

    def get_guides_agents(self) -> List[StateAgent]:
        agents = []
        for k, v in self.agents_by_breed.items():
            if issubclass(k, GuideAgent):
                agents.extend(list(v))

        return agents

    def get_agent_by_id(self, uid: int) -> StateAgent:
        return self._agents[uid]
