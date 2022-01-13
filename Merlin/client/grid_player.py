from .helper_classes import *
from Engine.client.unit import Unit
from game.constants import *
import queue
import copy
from collections import deque
import random
from typing import List, Tuple, Type, Union, Optional

from game.constants import *

from Engine.server.move import *

from itertools import combinations

class GridPlayer:

    def __init__(self):
        self.foo = True
        self.count = 0
        self.id_dict = dict()
        self.avail_resources = dict()
        self.enemy_resources = dict()
        self.dead_tasks = {1: [], 2: [], 3: [], 4: []}  # Tasks of the dead Units
        self.queue = deque()
        self.locked = []
        self.initialized = False

    def tick(self, game_map: Map, your_units: Units, enemy_units: Units, resources: int, turns_left: int, your_flag: dict,
             enemy_flag: dict):
        # Task, the player codes this
        delete, add = [], []
        if self.initialized is False:  # initalize all the stuff needed
            self.initialized = True
            half_grid = len(game_map.grid) // 2
            for i in game_map.find_all_resources():  # which resources are on our side
                if (i[1] > half_grid) ^ (your_flag['y'] > half_grid):
                    self.enemy_resources[i] = -1
                else:
                    self.avail_resources[i] = -1
            self.initialize_tags(your_units)  # take cares of new and delered units
            self.initialize_locked(game_map)  # which positions are always locked
            self.queue.append(Buy(Units.WORKER))
            self.queue.append(Buy(Units.WORKER))
            self.queue.append(Buy(Units.WORKER))
            self.queue.appendleft(GoToMine())
            self.queue.appendleft(GoToMine())
            self.queue.appendleft(GoToMine())
            self.queue.appendleft(GoToMine())
            self.queue.appendleft(GoToMine())
            self.queue.append(Buy(Units.WORKER))
        print("turn taken: ", self.count)
        print('universal q', self.queue)
        if self.initialized:
            add, delete = self.adding_tags(your_units)
        locked = self.update_locked(enemy_units)
        self.initialize_decision(your_units)
        for i in add:
            eoeo = self.queue.pop()
            print('popped from universal q: ', eoeo, 'for ', i.get_id())
            i.add_decision(eoeo)
        for j in self.id_dict.values():
            if j.available(your_units.get_unit(j.get_id())) and self.queue:
                j.add_decision(self.queue.pop())
        lst = [i.make_decision(your_units.get_unit(i.id), game_map=game_map, avail_resources=self.avail_resources,
                               locked=locked) for i in self.id_dict.values()]
        self.count += 1
        lst = list(filter((None).__ne__, lst))
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
        temp_dict = set(your_units.get_all_unit_ids()) - set(self.id_dict.keys())
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
    
    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return

    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return


class GameUnit:

    def __init__(self, idd: int):
        self.id = idd
        self.decision = Q()
        self.current = None

    def add_decision(self, decision: Decision) -> None:
        """
        Adds a decision to the self.decision queue.
        """
        self.decision.put(decision)
        if self.current is None:
            self.current = self.decision.get()
            #self.time = self.current.time

    def make_decision(self, unit: Unit, **kwargs) -> None:
        """
        Dequeues a decision and returns the corresponding move
        """
        if self.current.are_u_done(unit, **kwargs):
            self.current = self.decision.get(unit.type)
        return self.current.next_move(unit, **kwargs)

    def direction(self, locked: List[List[int]]) -> Direction:  # TODO #2 need to implement properly
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
        return self.id + ' ' + str(self.current)

    def available(self, unit) -> bool:
        return self.current is None or self.current.are_u_done(unit)


class Attack(Decision):

    def __init__(self, where: Tuple[int, int]):
        super().__init__()
        self.where = where

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        # TODO #1
        return

    def __str__(self) -> str:
        return 'Attack'


class Mine(Decision):

    def __init__(self):
        super().__init__()
        self.time = 2
        self.mined = False

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        if self.mined is False:
            self.mined = True
            return createMineMove(unit.id)
        self.time -= 1

    def __str__(self) -> str:
        return 'Mine'

    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return self.time <= 0

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
        return createDirectionMove(unit.id, direction_to(unit, self.path.pop(0)), MAX_MOVEMENT_SPEED[Units.WORKER])

    def reset(self):
        self.current = None
        self.path = None

    def __str__(self) -> str:
        return 'GoTo'


class Buy(Decision):

    def __init__(self, piece_type: Units):
        super().__init__()
        self.piece_type = piece_type
        self.bought = False
        self.time = 1

    def next_move(self, unit: Unit, **kwargs) -> Optional[Tuple[Moves, int, Type, Direction]]:
        locked = kwargs.get('locked')
        c = unit.x
        r = unit.y
        dir = None
        for i in ((c, r + 1),(c, r - 1), (c + 1, r), (c - 1, r)):
            if is_within_map(locked, i[0], i[1]) and locked[i[1]][i[0]] != 1:
                dir = direction_to(unit, i)
                break
        if dir is not None and self.bought is False:
            self.bought = True
            return createBuyMove(unit.id, self.piece_type, dir)
        self.time -= 1
    
    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return self.time <= 0

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
        self.time = 2

    def next_move(self, unit: Unit, **kwargs) -> Union[Tuple[Moves, int, Direction, int], Tuple[Moves, int]]:
        if self.destination is None:
            self.destination = closest_resource(kwargs.get('avail_resources'), unit)
            kwargs.get('avail_resources')[self.destination] = unit.id
        if (unit.x, unit.y) != self.destination:
            path = bfs(kwargs.get('locked'), (unit.x, unit.y), self.destination)
            print(path)
            next_pos = path[1]
            kwargs.get('locked')[next_pos[1]][next_pos[0]] = 1
            return createDirectionMove(unit.id, direction_to(unit, next_pos), MAX_MOVEMENT_SPEED[Units.WORKER])
        if (unit.x, unit.y) == self.destination and self.mined is False:
            self.mined = True
            self.time -= 1
            return createMineMove(unit.id)
        if self.mined is True:
            self.time -= 1

    def are_u_done(self, unit: Unit, **kwargs):
        return (unit.x, unit.y) == self.destination and self.time <= 0
            

    def reset(self):
        self.current = None
        self.path = None

    def __str__(self) -> str:
        return 'GoToMine'


class Q:

    def __init__(self):
        self.queue = queue.Queue()

    def get(self, unit_type: Type = None) -> Decision:
        """
        Returns a Mine decision if id is given otherwise pops the latest decision in <self.queue> and returns it.
        """
        if self.queue.empty() and unit_type == Units.WORKER:
            return Mine()
        return self.queue.get()

    def put(self, item: Decision) -> None:
        """
        Adds a decision to the <self.queue>.
        """
        self.queue.put(item)

    def empty(self, type=None) -> bool:
        """
        Returns False if <type> given is Worker otherwise returns if <self.queue> is empty.
        """
        if type == Units.WORKER:
            return False
        return self.queue.empty()
    
    def __str__(self) -> str:
        return str(self.k)

    def __str__(self) -> str:
        return str(self.k)


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
        dc = c_2 - c
        dr = r_2 - r
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
        dir = [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        random.shuffle(dir)
        for adj in dir:
            if is_within_map(graph, adj[0], adj[1]) and (graph[adj[1]][adj[0]] != '1') and adj not in vis:
                queue.append(path + [adj])
                vis.add(adj)


def is_within_map(map: List[List[int]], x: int, y: int) -> bool:
    """
    Returns if the coordinate (<x>, <y>) is in <map>.
    """

    return 0 <= x < len(map[0]) and 0 <= y < len(map)

def bipartite_graph_min_weight(source: list[Tuple[int, int, int]], target: list[Tuple[int, int]]) -> dict[int: Tuple[int, int]]:
    all_combinations = combinations(target, len(source))
    temp = []
    for target_combo in all_combinations:
        lst = []
        for index, source_combo in enumerate(source):
            dist = abs(source_combo[1]-target_combo[index][1]) + abs(source_combo[0] - target_combo[index][0])
            lst.append((source_combo[2], dist, target_combo[index])) # (id, distance, target)
        temp.append(lst)
    min_combo = min(temp, key= lambda x: sum(i[1] for i in x))
    min_dict = {i[0]: i[2] for i in min_combo}
    return min_dict