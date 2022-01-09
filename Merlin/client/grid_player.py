from .helper_classes import *
from Engine.client.unit import Unit
from game.constants import *
import queue
import copy

from typing import List, Tuple, Type, Union, Optional

from game.constants import *

from Engine.server.move import *


class GridPlayer:

    def __init__(self):
        self.foo = True
        self.count = 0
        self.id_dict = dict()
        self.avail_resources = dict()
        self.enemy_resources = dict()
        self.dead_tasks = {1: [], 2: [], 3: [], 4: []}  # Tasks of the dead Units
        self.queue = queue.Queue()
        self.locked = []

    def tick(self, game_map, your_units: Units, enemy_units: List[Unit], resources, turns_left: int, your_flag,
             enemy_flag):
        # Task, the player codes this
        delete, add = [], []
        if self.count <= 1:  # initalize all the stuff needed
            half_grid = len(game_map.grid) // 2
            for i in game_map.find_all_resources():  # which resources are on our side
                if (i[1] > half_grid) ^ (your_flag['y'] > half_grid):
                    self.enemy_resources[i] = -1
                else:
                    self.avail_resources[i] = -1
            self.initialize_tags(your_units)  # take cares of new and delered units
            self.initialize_locked(game_map)  # which positions are always locked
            add = self.id_dict.keys()
        locked = self.update_locked(enemy_units)
        self.initialize_decision(your_units)
        print("turn taken: ", self.count)
        for i in self.avail_resources:
            self.queue.put(GoToMine())
        # start = []
        # for i in your_units.get_all_unit_of_type(Units.WORKER):
        #     if self.id_dict[str(i.id)].time <= 0:               # i.id is integer but function returns as str
        #         start.append((i.x, i.y, str(i.id)))
        # # start = [self.id_dict[i.id] for i in your_units.get_all_unit_of_type(Units.WORKER) if self.id_dict[i.id].time <= 0]
        # closest = closest_pairs(start, self.avail_resources)
        #for i in your_units.get_all_unit_of_type(Units.WORKER):
        print(self.id_dict,type(your_units),add)
        for i in add:
            self.id_dict[i].add_decision(self.queue.get())
        lst = [i.make_decision(your_units.get_unit(i.id), game_map = game_map, avail_resources = self.avail_resources, locked = locked) for i in self.id_dict.values()]
        # for i in closest:
        #     self.id_dict[i[2]].add_decision(GoToMine((i[0], i[1])))
        #     lst.append(self.id_dict[i[2]].make_decision(your_units.get_unit(i[2]), bfs= game_map.bfs))
        self.count += 1
        print(lst)
        return lst

    def initialize_locked(self, game_map: Map) -> None:
        """
        Copies the game map to a list.
        Elements of list are 1 if position is blocked and 0 otherwise.
        """
        for i in game_map.grid:
            self.locked.append([1 if j == 'X' else 0 for j in i])

    def update_locked(self, enemy_units: List[Unit]) -> List[List[int]]:
        """
        Updates the map with the enemy positions and sets positions as 1 (blocked) and returns the updated list.
        """
        locked = copy.deepcopy(self.locked)
        for i in enemy_units.units:
            locked[i.x][i.y] = 1
        return locked

    def initialize_decision(self, your_units: List[Unit]) -> None:
        """
        Update self.id_dict by adding new Units and removing deleted (killed) Units.
        Take deleted Units tasks and update self.dead_tasks (adding their tasks).
        """
        add, delete = self.adding_tags(your_units)
        for i in delete:
            temp = self.id_dict.pop(i)
            if temp.time > 0:
                temp.decision.put(temp.current)
            self.dead_tasks[temp.type].append(temp.decision)

        for i in add:
            temp = self.id_dict[i]
            if self.dead_tasks[temp.type]:
                temp.decision = self.dead_tasks[temp.type].pop()

    def adding_tags(self, your_units: List[Unit]) -> Tuple[List[Unit], List[Unit]]:
        """
        takes systems ids, map them to our own id system, if it has some new ids, get units info,
        assign them new ids in our system. If its a new ID, give a custom new ID in our system as well.
        If ID is not seen on map then clear ID from hashmap ( worker has been destroyed ).
        Returns all the GameUnits that have been deleted and all GameUnits that have been created
        """
        temp_dict = set(your_units.get_all_unit_ids()) - set(self.id_dict)
        lst1 = list()  # New units (on the board)
        for i in temp_dict:
            game_unit = GameUnit(i)
            self.id_dict[i] = game_unit
            lst1.append(game_unit)
        lst = list()  # Deleted units (not on the board)
        temp = set(self.id_dict) - set(your_units.get_all_unit_ids())
        for i in temp:
            lst.append(self.id_dict.pop(i))

        return lst1, lst

    def initialize_tags(self, your_units: List[Unit]) -> None:
        """
        Initialize self.id dict with their corresponding Unit
        """
        lst_unit_ids = your_units.get_all_unit_ids()

        self.id_dict = {i: GameUnit(i) for i in lst_unit_ids}

    def first_phase(self, lst_workers):
        """
        """


class Decision:

    def __init__(self) -> None:
        """"""
        self.time = 1

    def next_move(self, unit: Unit, **Kwargs) -> Tuple[Move, int]:
        """
        Returns some move based on the game decision.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the decisions so the decision is reusable.
        """
        return

    def __str__(self) -> str:
        raise NotImplementedError


class GameUnit:

    def __init__(self, idd: int):
        self.id = idd
        self.decision = queue.Queue()
        self.time = 0
        self.current = None

    def add_decision(self, decision: Decision) -> None:
        """
        Adds a decision to the self.decision queue.
        """
        self.decision.put(decision)

    def make_decision(self, unit: Unit, **kwargs) -> None:
        """
        Dequeues a decision and returns the corresponding move
        """
        print('h')
        if self.time <= 0 and not self.decision.empty():
            self.current = self.decision.get()
            next_move = self.current.next_move(unit, **kwargs)
            self.time = self.current.time
        elif self.time > 0 and self.current:
            next_move = self.current.next_move(unit, **kwargs)
        else:
            self.current = None
            next_move = None
        self.time -= 1
        return next_move

    def direction(self) -> Direction:  # need to implement properly
        """
        Returns an empty location direction adjacent to Unit's position.
        """
        return Direction.DOWN

    def get_id(self) -> int:
        """
        Getter for self.id
        """
        return self.id

    def reset(self) -> None:
        """
        Resets the class
        """
        self.id = -1
        if self.time > 0:
            self.decision.put(self.current)
        self.time = 0

    def __str__(self) -> str:
        return self.id + ' ' + str(self.time) + ' ' + str(self.current)

    def available(self) -> bool:
        return self.time <= 0

class Attack(Decision):

    def __init__(self, where: Tuple[int, int]):
        super().__init__()
        self.where = where

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        return

    def __str__(self) -> str:
        return 'Attack'

class Mine(Decision):

    def init(self):
        super().__init__()
        self.time = 2
        self.mined = False

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        if self.mined is False:
            self.mined = True
            return createMineMove(unit.id)

    def __str__(self) -> str:
        return 'Mine'


class GoTo(Decision):

    def __init__(self, destination: Tuple[int, int]):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        if self.path is None:
            self.current = (unit.x, unit.y)
            self.path = kwargs.get('bfs')(self.current, self.destination)
            self.time = len(set(self.path) - set(self.current))
            self.path.pop(0)
        return createDirectionMove(unit.id, direction_to(unit,self.path.pop(0)), MAX_MOVEMENT_SPEED[Units.WORKER])

    def reset(self):
        self.current = None
        self.path = None

    def __str__(self) -> str:
        return 'GoTo'


class Buy(Decision):

    def __init__(self, piece_type: Type, direction: Direction):
        super().__init__()
        self.piece_type = piece_type
        self.direction = direction

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Type, Direction]:
        return createBuyMove(unit.id, unit.type, unit.direction())

    def __str__(self) -> str:
        return 'Buy'


class Upgrade(Decision):

    def __init__(self):
        super().__init__()

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        return createUpgradeMove(unit.id)

    def __str__(self) -> str:
        return 'Upgrade'

class GoToMine(Decision):

    def __init__(self, destination: Tuple[int, int] = None):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None
        self.mined = False

    def next_move(self, unit: Unit, **kwargs) -> Union[Tuple[Moves, int, Direction, int], Tuple[Moves, int]]:
        if self.destination is None:
            print('h')
            self.destination = closest_resource(kwargs.get('avail_resources'), unit)

            kwargs.get('avail_resources')[self.destination] = unit.id
            print(self.destination)
        if self.path is None and self.destination is not None:
            self.current = (unit.x, unit.y)
            self.path = bfs(kwargs.get('locked'), self.current, self.destination)
            self.path.pop(0)
            self.time = len(set(self.path) - set(self.current)) + 2

        if (len(self.path) == 0 or self.destination is None) and self.mined is False:
            self.mined = True
            return createMineMove(unit.id)
        if len(self.path) != 0:
            next_pos = self.path.pop(0)
            kwargs.get('locked')[next_pos[1]][next_pos[0]] = 1
            return createDirectionMove(unit.id, direction_to(unit,next_pos), MAX_MOVEMENT_SPEED[Units.WORKER])

    def reset(self):
        self.current = None
        self.path = None

    def __str__(self) -> str:
        return 'GoToMine'


def closest_resource(avail_resources: dict, unit: Unit) -> Optional[Tuple[int, int]]:
    """
    Takes the <avail_resources> and the <unit> and returns the closest resource available for <unit>
    """
    locations = [i for i in avail_resources if avail_resources[i] == -1]
    if not locations:
        return None
    c, r = unit.position()
    result = None
    so_far = 999999
    for (c_2, r_2) in locations:
        dc = c_2-c
        dr = r_2-r
        dist = abs(dc) + abs(dr)
        if dist < so_far:
            result = (c_2, r_2)
            so_far = dist
    return result


def bfs(grid: List[List[int]], start: Tuple[int, int], dest: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        (Map, (int, int), (int, int)) -> [(int, int)]
        ### tuples are col, row , col,row
        Finds the shortest path from <start> to <dest>.
        Returns a path with a list of coordinates starting with
        <start> to <dest>.

        <grid> is the locked position.
        """
        graph = grid
        queue = [[start]]
        vis = set(start)
        if start == dest or graph[start[1]][start[0]] == '1' or \
                not (0 < start[0] < len(graph[0])-1
                    and 0 < start[1] < len(graph)-1):
            return None
        while queue:
            path = queue.pop(0)
            node = path[-1]

            r = node[1]
            c = node[0]
            if node == dest:
                return path
            for adj in ((c+1, r), (c-1, r), (c, r+1), (c, r-1)):
                if is_within_map( graph, adj[0],adj[1]) and (graph[adj[1]][adj[0]]  != '1') and adj not in vis:
                    queue.append(path + [adj])
                    vis.add(adj)


def is_within_map(map: List[List[int]], x: int, y: int) -> bool:
    """
    Returns if the coordinate (<x>, <y>) is in <map>.
    """

    return 0 <= x < len(map[0]) and 0 <= y < len(map)