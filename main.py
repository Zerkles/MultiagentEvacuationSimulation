import json
import os

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider, Checkbox, Choice, StaticText

from agents.agents import Evacuee, GuideAgent, Exit, Obstacle, MapInfo
from simulation.model import EvacuationModel


def agents_portrayal(agent):
    if agent is None:
        return

    portrayal = dict()

    if type(agent) is Evacuee:
        portrayal["Color"] = "red"
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["r"] = 1

    elif isinstance(agent, GuideAgent):
        portrayal["Color"] = "blue"
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["r"] = 1

    elif type(agent) is Exit:
        portrayal["Color"] = "green"
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is Obstacle:
        portrayal["Color"] = "grey"
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1
    elif type(agent) is MapInfo:
        portrayal["Color"] = agent.color
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Layer"] = 2
        portrayal["text"] = str(agent.value)
        portrayal["text_color"] = "black"

    return portrayal


HEIGHT = WIDTH = 100

canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 950, 550)
# canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 1200, 1200)
chart_element = ChartModule([{"Label": "Evacuees", "Color": "#AA0000"}])

model_params = {
    "width": WIDTH,
    "height": HEIGHT,

    "General Info": StaticText("General settings"),
    "ghost_agents": Checkbox("Ghost agents", False),
    "show_map": Checkbox("Show distance for each field", False),

    "Guides Info": StaticText("Guides settings:"),
    "guides_random_position": Checkbox("Guides start from random position", False),
    "guides_num": Slider("Number of Guides", 2, 1, 4),
    "guides_mode": Choice("Guides action mode", "Q Learning", ["A", "B", "None", "Q Learning"]),

    "Evacuees Info": StaticText("Evacuees settings:"),
    "evacuees_share_information": Checkbox("Evacuees share exit area information", False),
    "evacuees_num": Slider("Number of Evacuees", 500, 1, 2000),

    "Map Info": StaticText("Map settings:"),
    "map_type": Choice("Map type", 'default', ["default", "cross", "boxes", 'random_rectangles']),

    "Cross Map Info": StaticText("Only for cross map:"),
    "cross_gap": Slider("Cross gap (symmetric)", 10, 1, 25),

    "Boxes Map Info": StaticText("Only for boxes map:"),
    "boxes_thickness": Slider("Boxes thickness", 15, 1, 20),

    "Random Rectangles Map Info": StaticText("Only for random_rectangles map:"),
    "rectangles_num": Slider("Number of rectangles", 10, 1, 30),
    "rectangles_max_size": Slider("Maximal length of one side", 15, 1, 50),
    "erosion_proba": Slider("Probability for each position to disappear", 0.5, 0.0, 1.0, 0.1),

    "qlearning_params": {'epsilon': 0.0, 'gamma': 0.8, 'alpha': 0.0, 'weights': None},
    "extractor_maps": None,
}
if __name__ == '__main__':
    if os.path.exists("output/weights_visited.txt"):
        with open("output/weights_visited.txt", "r") as f:
            model_params['qlearning_params']['weights'] = dict(json.load(f))

    server = ModularServer(EvacuationModel, [canvas_element, chart_element], "Multiagent Evacuation Simulation",
                           model_params)
    server.port = 8521
    server.launch()
