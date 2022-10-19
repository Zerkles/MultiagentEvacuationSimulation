from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider, Checkbox, Choice

from agents import Evacuee, Guide, Obstacle, Exit, MapInfo
from model import EvacuationModel


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

    elif type(agent) is Guide:
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

# canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 900, 500)
canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 1500, 1500)
chart_element = ChartModule([{"Label": "Evacuees", "Color": "#AA0000"}])

model_params = {
    "width": WIDTH,
    "height": HEIGHT,
    "map_type": Choice("Map type", "cross", ["default", "cross", "boxes"]),
    "guides_mode": Choice("Guides action mode", "B", ["A", "B"],
                          description="Guides has different action scheme."),
    "evacuees_num": Slider("Number of Evacuees ", 500, 1, 2000),
    "guides_num": Slider("Number of Guides", 2, 1, 4),
    "ghost_agents": Checkbox("Ghost agents", False,
                             description="Multiple agents can stand at one position."),
    "evacuees_share_information": Checkbox("Evacuees share exit area information", False),
    "guides_random_position": Checkbox("Guides start from random position", False,
                                       description="If set to false, guides starts from fixed positions (same as location of points)."),

}

server = ModularServer(EvacuationModel, [canvas_element, chart_element], "Multiagent Evacuation Simulation",
                       model_params)

server.port = 8521
server.launch()
