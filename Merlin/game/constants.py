from enum import Enum

TURNS_PER_PLAYER = 200


# Represents all the valid moves a player can make
class Moves(Enum):
    ATTACK = 1
    UPGRADE = 2
    DIRECTION = 3
    MINE = 4
    BUY = 5
    CAPTURE = 6


# Represents all units in the game
class Units(Enum):
    WORKER = 1
    SCOUT = 2
    KNIGHT = 3
    ARCHER = 4


class Tiles():
    GROUND = ' '  # A walkable tile
    WALL = 'X'  # A non-walkable tile
    GOLD = 'G'  # A gold resource tile, check manual for worth when mined.
    SILVER = 'S'  # A silver resource tile, check manual for worth when mined.
    COPPER = 'C'  # A copper resource tile, check manual for worth when mined.
    BASE = 'B'  # A team's base


# Direction holds string values for each cardinal direction on the board
class Direction(Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'


# 0 represents cost to buy, anything above represents cost to upgrade
UPGRADE_COSTS = {
    Units.WORKER: {
        0: 10,
        1: 10,
        2: 10,
        3: 10
    },
    Units.SCOUT: {
        0: 10,
        1: 20,
        2: 20,
        3: 20
    },
    Units.KNIGHT: {
        0: 10,
        1: 20,
        2: 20,
        3: 20
    },
    Units.ARCHER: {
        0: 10,
        1: 20,
        2: 20,
        3: 20
    }
}
# unittype to
ATTACK_DAMAGE = {
    Units.WORKER: {
        1: 0,
        2: 0,
        3: 0
    },
    Units.SCOUT: {
        1: 0,
        2: 0,
        3: 0
    },
    Units.KNIGHT: {
        1: 50,
        2: 80,
        3: 100
    },
    Units.ARCHER: {
        1: 20,
        2: 50,
        3: 80
    }
}

MAX_LEVEL = 3

MAX_ATTACK_RANGE = {
    Units.ARCHER: 2,
    Units.WORKER: 0,
    Units.SCOUT: 1,
    Units.KNIGHT: 1
}

MAX_MOVEMENT_SPEED = {
    Units.ARCHER: 1,
    Units.WORKER: 1,
    Units.SCOUT: 1,
    Units.KNIGHT: 2
}

MINING_REWARDS = {
    Tiles.GOLD: 75,
    Tiles.SILVER: 50,
    Tiles.COPPER: 25
}

MINING_TIME = 3  # One of your turns, one of the enemies turns.
DUPLICATION_TIME = 3  # One of your turns, one of the enemies turns.
