from enum import Enum


class Terrains(Enum):
    """
    Defined for readability, to be moved to different script and used also for simulator interface
    Strings were chosen as decoding these concatenated with standard prompt gives us a nice flag token in train logic
    """
    LEFT_WALL = "L"
    RIGHT_WALL = "R"
    BUMPY = "B"
    TUNNEL = "T"