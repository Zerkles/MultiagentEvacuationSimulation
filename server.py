from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider, Checkbox, Choice

from agents import Evacuee, Guide, Obstacle, Exit
from model import EvacuationModel


def agents_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Evacuee:
        portrayal["Color"] = "red"
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1

    elif type(agent) is Guide:
        portrayal["Color"] = "blue"
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
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

    return portrayal


canvas_element = CanvasGrid(agents_portrayal, 100, 100, 800, 800)
chart_element = ChartModule(
    [{"Label": "Evacuees", "Color": "#AA0000"}, {"Label": "Guides", "Color": "#AA0000"}]
)

model_params = {
    "map_type": Slider("Map type (Not Implemented)", 1, 1, 3),
    "evacuees_num": Slider("Number of Evacuees (Not Implemented)", 50, 1, 100),
    "guides_num": Slider("Number of Guides (Not Implemented)", 2, 1, 8),
    "guides_mode": Slider("Guides action mode (Not Implemented)", 1, 1, 3),
    "agents_clipping": Checkbox("Agents are clipping (Not Implemented)", True),
    "evacuees_random_position": Checkbox("Evacuees start from random position (Not Implemented)", True),
    "guides_random_position": Checkbox("Guides start from random position (Not Implemented)", True),
}

server = ModularServer(
    EvacuationModel, [canvas_element, chart_element], "Multiagent Evacuation Simulation", model_params
)
server.port = 8521
server.launch()
