import json
from typing import List

from Engine.client.unit import Unit
from .client_connection import ClientConnection
from copy import deepcopy
from .maps import Map

class GridGame():
    """
    GridGame is the currently running game, it controls all game state and updates the state each turn with tick.
    """

    def __init__(self, player_one_connection:ClientConnection, player_two_connection:ClientConnection, gamemap:Map):
        self.next_id = 0
        self.all_units = {}
        self.p1_conn = player_one_connection
        self.p2_conn = player_two_connection
        self.p1_units = {}
        self.p2_units = {}
        self.resources = {
            self.p1_conn.name: 100,
            self.p2_conn.name: 100}
        #TODO resources must be removed?
        #Creates 2 copies of the map, one reversed of the other
        top = gamemap.map
        bottom = deepcopy(top[:-1])
        bottom.reverse()
        #Creates the map by combining the top and bottom copies of the map and adding the units to the game state.
        self.grid = top + bottom
        self.current_player_turn = 0

    #add_unit gives the unit a id and adds the unit to the games state
    def add_unit(self, player, unit):
        unit.id = self.next_id
        player[str(self.next_id)] = unit
        self.next_id += 1
        self.all_units['{},{}'.format(unit.x, unit.y)] = unit

    def get_tile(self, unit:Unit):
        x,y = unit.pos_tuple()
        return self.grid[y][x]

    def move_unit(self, x, y, unit):
        del self.all_units['{},{}'.format(x, y)]
        self.all_units['{},{}'.format(unit.x, unit.y)] = unit

    def get_unit(self, x, y):
        return self.all_units['{},{}'.format(x, y)]

    def has_unit(self, x, y):
        return '{},{}'.format(x, y) in self.all_units

    def del_unit(self, x, y):
        unit = self.all_units['{},{}'.format(x, y)]
        del self.all_units['{},{}'.format(x, y)]

    # returns the player to which the unit is owned by.
    def get_unit_player(self, unit):
        if str(unit.id) in self.p1_units.keys():
            return self.p1_units
        else:
            return self.p2_units

    def get_unit_player_name(self, unit):
        if str(unit.id) in self.p1_units:
            return self.p1_conn.name
        else:
            return self.p2_conn.name

    def json_str(self):
        display = deepcopy(self.grid)
        for u in self.p1_units.values():
            display[u.y][u.x] = u
        for u in self.p2_units.values():
            display[u.y][u.x] = u.string().upper()

        def inner(r): return '[{}]'.format(
            ','.join(map(lambda x: (x if isinstance(x, str) else x.string()), r)))
        return '[{}]'.format(','.join(map(inner, display)))

    def print_map(self, p1_name, p2_name):
        j = json.dumps({
            'map': self.json_str(),
            'p1_resources': self.resources[p1_name],
            'p2_resources': self.resources[p2_name]
        })
        print('MAP:{}'.format(j))

    def get_state(self):
        return self.grid, self.all_units, {**self.resources}

    def verify_move(self, k, v, player_state, player_resources, name, enemy_units, **args):
        raise NotImplementedError

    def make_move(self, k, v, player_state, player_name, opponent_state):
        raise NotImplementedError

    def getWinner(self):
        raise NotImplementedError("Must define a wincondition")

    def tick_player(self, conn, current, opponent, name, turns):
        raise NotImplementedError("Must define a player's tick")

    #tick is run next turn and updates the game state
    def tick(self):
        raise NotImplementedError

class GridGameFactory():
    def getGame(self, connections:List[ClientConnection], map:Map)->GridGame:
        raise NotImplementedError
    def serialize_unit(self, unit:Unit):
        raise NotImplementedError
    def deserialize_unit(self, unit:dict):
        raise NotImplementedError