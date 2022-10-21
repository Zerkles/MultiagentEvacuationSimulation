from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider, Checkbox, Choice, StaticText

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

canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 1100, 750)
# canvas_element = CanvasGrid(agents_portrayal, WIDTH, HEIGHT, 1500, 1500)
chart_element = ChartModule([{"Label": "Evacuees", "Color": "#AA0000"}])

model_params = {
    "width": WIDTH,
    "height": HEIGHT,

    "General Info": StaticText("General settings"),
    "ghost_agents": Checkbox("Ghost agents", False),
    "show_map": Checkbox("Show distance for each field", False),

    "Guides Info": StaticText("Guides settings:"),
    "guides_num": Slider("Number of Guides", 2, 1, 4),
    "guides_mode": Choice("Guides action mode", "none", ["A", "B", "none"]),
    "guides_random_position": Checkbox("Guides start from random position", False),

    "Evacuees Info": StaticText("Evacuees settings:"),
    "evacuees_num": Slider("Number of Evacuees", 500, 1, 2000),
    "evacuees_share_information": Checkbox("Evacuees share exit area information", False),

    "Map Info": StaticText("Map settings:"),
    "map_type": Choice("Map type", 'default', ["default", "cross", "boxes", 'random_rectangles']),

    "Random Rectangles Info": StaticText("Only for random_rectangles map:"),
    "rectangles_num": Slider("Number of rectangles", 10, 1, 30),
    "rectangles_max_size": Slider("Maximal length of one side", 15, 1, 50),
    "erosion_proba": Slider("Probability for each position to disappear", 0.5, 0.0, 1.0, 0.1),
}

server = ModularServer(EvacuationModel, [canvas_element, chart_element], "Multiagent Evacuation Simulation",
                       model_params)

server.port = 8521
server.launch()
# TODO: Change default FPS to 0 (Max possible)
