from .move import Move
from game.constants import Tiles, Moves
from Engine.client.unit import Unit


def createAttackMove(unitId, direction, length):
    '''
    This is a helper function used by the player to ATTACK an enemy unit

    Parameters:
    unitId - The ID of the unit that's attacking
    direction - The direction to attack in, must be one of the values from the Direction enum (game.constants line 29)
    length - how far the unit is attacking, must be less than the unit's MAX_ATTACK_RANGE (game.constants line 89)
    '''

    return (Moves.ATTACK, unitId, direction, length)


def createUpgradeMove(unitId):
    '''
    This is a helper function used by the player to UPGRADE a unit

    Parameters:
    unitId - The ID of the unit that is being upgraded
    '''
    return (Moves.UPGRADE, unitId)


def createDirectionMove(unitId, direction, magnitude):
    '''
    This is a helper function used by the player to MOVE a unit

    Parameters:
    unitId - The ID of the unit that is moving
    direction - The direction to move in, must be one of the values from the Direction enum (game.constants line 29)
    magnitude - how far the unit is moving, must be less than the unit's MAX_MOVEMENT_SPEED (game.constants line 96)
    '''

    return (Moves.DIRECTION, unitId, direction, magnitude)


def createMineMove(unitId):
    '''
    This is a helper function used by the player to MINE a resource, NOTE: Only workers can mine! Must be on top of the resource to mine

    Parameters:
    unitId - The ID of the unit that is mining
    '''

    return (Moves.MINE, unitId)


def createBuyMove(unitId, unitType, direction):
    '''
    This is a helper function used by the player to BUY a new unit, NOTE: Only workers can buy units!

    Parameters:
    unitId - The ID of the unit that's buying the new unit
    direction - The direction to place the new unit in, must be one of the values from the Direction enum (game.constants line 29)
    '''

    return (Moves.BUY, unitId, unitType, direction)


def createCaptureMove(unitId, direction):
    '''
    This is a helper function used by the player to CAPTURE a flag, NOTE: Only Scouts can capture flags! Must be beside the flag to capture

    Parameters:
    unitId - The ID of the unit that's capturing
    direction - The direction to capture in, must be one of the values from the Direction enum (game.constants line 29)
    '''

    return (Moves.CAPTURE, unitId, direction)


class Map:
    # all outputs will be of the form (x, y). i.e., (c, r).
    def __init__(self, map_grid: [[str]]) -> None:
        """
        Initialize a new Map.
        """
        self.grid = map_grid

    def get_tile(self, x: int, y: int) -> str:
        """
        Returns the tile found at <x> and <y>.
        Preconditions: x >= 0
                       y >= 0
        """
        return self.grid[y][x]

    def is_wall(self, x: int, y: int) -> bool:
        """
        Returns whether the tile at <x> and <y> is a wall.
        Preconditions: x >= 0
                       y >= 0
        """
        return self.grid[y][x].upper() == Tiles.WALL

    def is_resource(self, x: int, y: int) -> bool:
        """
        Returns whether the tile at <x> and <y> is a resource.
        Preconditions: x >= 0
                       y >= 0
        """
        return self.grid[y][x].upper() in [Tiles.COPPER, Tiles.SILVER, Tiles.GOLD]

    def find_all_resources(self) -> [(int, int)]:
        """
        Returns the (x, y) coordinates for all resource nodes.
        """
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.is_resource(col, row):
                    locations.append((col, row))
        return locations

    def closest_resources(self, unit: Unit) -> (int, int):
        """
        Returns the coordinates of the closest resource to <unit>.
        """
        locations = self.find_all_resources()
        c, r = unit.position()
        result = None
        so_far = 999999
        for (c_2, r_2) in locations:
            dc = c_2 - c
            dr = r_2 - r
            dist = abs(dc) + abs(dr)
            if dist < so_far:
                result = (c_2, r_2)
                so_far = dist
        return result

    def bfs(self, start: (int, int), dest: (int, int)) -> [(int, int)]:
        """(Map, (int, int), (int, int)) -> [(int, int)]
        Finds the shortest path from <start> to <dest>.
        Returns a path with a list of coordinates starting with
        <start> to <dest>.
        """
        graph = self.grid
        queue = [[start]]
        vis = set(start)
        if start == dest or graph[start[1]][start[0]] == 'X' or \
                not (0 < start[0] < len(graph[0]) - 1
                     and 0 < start[1] < len(graph) - 1):
            return None

        while queue:
            path = queue.pop(0)
            node = path[-1]
            r = node[1]
            c = node[0]

            if node == dest:
                return path
            for adj in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if (graph[adj[1]][adj[0]] == ' ' or
                    graph[adj[1]][adj[0]] == 'R') and adj not in vis:
                    queue.append(path + [adj])
                    vis.add(adj)


class Units:
    def __init__(self, units: dict) -> None:
        """
        Initialize a new Units.
        """
        self.units = {}  # a dictionary of unit objects.
        for unit in units:
            self.units[str(unit['id'])] = Unit(unit)

    def get_unit(self, id: str) -> Unit:
        """
        Return the Unit with <id>.
        """
        return self.units[id]

    def get_all_unit_ids(self) -> [str]:
        """
        Returns the id of all current units.
        """
        all_units_ids = []
        for id in self.units:
            all_units_ids.append(id)
        return all_units_ids

    def get_all_unit_of_type(self, type: str) -> [Unit]:
        """
        Returns a list of unit objects of a given type.
        """
        all_units = []
        for id in self.units:
            if self.units[id].type == type:
                all_units.append(self.units[id])
        return all_units


def coordinate_from_direction(x: int, y: int, direction: str) -> (int, int):
    """
    Returns the resulting (x, y) coordinates after moving in a
    direction> from <x> and <y>.
    Acceptable directions:
        'LEFT'
        'RIGHT'
        'UP'
        'DOWN'
    """
    if direction == 'LEFT':
        return (x - 1, y)
    if direction == 'RIGHT':
        return (x + 1, y)
    if direction == 'UP':
        return (x, y - 1)
    if direction == 'DOWN':
        return (x, y + 1)
