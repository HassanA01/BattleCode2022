from .helper_classes import *
from Engine.client.unit import Unit
from game.constants import *
import queue
import copy

from typing import List, Tuple, Type, Union

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

    def tick(self, game_map, your_units: List[Unit], enemy_units: List[Unit], resources, turns_left: int, your_flag,
             enemy_flag):
        # Task, the player codes this
        if self.count <= 1:  # initalize all the stuff needed
            half_grid = len(game_map.grid) // 2
            for i in game_map.find_all_resources():  # which resources are on our side
                if (i[1] > half_grid) ^ (your_flag['y'] > half_grid):
                    self.enemy_resources[i] = -1
                else:
                    self.avail_resources[i] = -1
            self.initialize_tags(your_units)  # take cares of new and delered units
            self.initialize_locked(game_map)  # which positions are always locked
        locked = self.update_locked(enemy_units)
        self.initialize_decision(your_units)
        print("turn taken: ", self.count)
        for i in self.avail_resources:
            self.queue.put(GoToMine(i))
        start = []
        for i in your_units.get_all_unit_of_type(Units.WORKER):
            if self.id_dict[str(i.id)].time <= 0:               # i.id is integer but function returns as str
                start.append((i.x, i.y, str(i.id)))
        # start = [self.id_dict[i.id] for i in your_units.get_all_unit_of_type(Units.WORKER) if self.id_dict[i.id].time <= 0]
        closest = closest_pairs(start, self.avail_resources)
        lst = []
        for i in closest:
            self.id_dict[i[2]].add_decision(GoToMine((i[0], i[1])))
            lst.append(self.id_dict[i[2]].make_decision(your_units.get_unit(i[2]), bfs= game_map.bfs))
        self.count += 1
        print(lst)
        return lst

    def initialize_locked(self, game_map: Map) -> None:
        for i in game_map.grid:
            self.locked.append([1 if j == 'X' else 0 for j in i])

    def update_locked(self, enemy_units: List[Unit]) -> List[List[int]]:
        locked = copy.deepcopy(self.locked)
        for i in enemy_units.units:
            locked[i.x][i.y] = 1
        return locked

    def initialize_decision(self, your_units: List[Unit]) -> None:
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

    def closest_resources(self, unit: Unit) -> Tuple[int, int]:
        """
        Returns the coordinates of the closest resource to <unit>.
        """
        c, r = unit.position()
        return min(abs(c_2 - c) + abs(r_2 - r) for (c_2, r_2) in
                   self.avail_resources.keys()) if self.avail_resources else -1, -1

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
        raise NotImplementedError

    def reset(self) -> None:
        return


class GameUnit:

    def __init__(self, idd: int):
        self.id = idd
        self.decision = queue.Queue()
        self.time = 0
        self.current = None

    def add_decision(self, decision: Decision) -> None:
        self.decision.put(decision)

    def make_decision(self, unit: Unit, **kwargs) -> None:
        """
        Dequeues a decision and returns the corresponding move
        """
        if self.time <= 0 and not self.decision.empty():
            self.current = self.decision.get()
            print(kwargs)
            next_move = self.current.next_move(unit, **kwargs)
            print("hello")
            self.time = self.current.time
        else:
            self.current = None
            next_move = None
        self.time -= 1
        return next_move

    def direction(self) -> 'str':  # need to implement properly
        return Direction.DOWN

    def get_id(self) -> int:
        return self.id

    def reset(self) -> None:
        self.id = -1
        if self.time > 0:
            self.decision.put(self.current)
        self.time = 0


class Attack(Decision):

    def __init__(self, where: Tuple[int, int]):
        super().__init__()
        self.where = where

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        return


class Mine(Decision):

    def init(self):
        super().__init__()
        self.time = 2

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        return createMineMove(unit.id)


class GoTo(Decision):

    def __init__(self, destination: Tuple[int, int]):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        """
        Tells us what the next move of this unit is.
        """
        if self.path is None:
            self.current = (unit.x, unit.y)
            self.path = kwargs.get('bfs')(self.current, self.destination)
            self.time = len(set(self.path) - set(self.current))
            self.path.pop(0)
        if self.path[0] != self.destination:
            return createDirectionMove(unit.id, unit.position(self.path.pop(0)), MAX_MOVEMENT_SPEED[Unit.WORKER])

    def reset(self):
        self.current = None
        self.path = None


class Buy(Decision):

    def __init__(self, piece_type: Type, direction: Direction):
        super().__init__()
        self.piece_type = piece_type
        self.direction = direction

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Type, Direction]:
        return createBuyMove(unit.id, unit.type, unit.direction())


class Upgrade(Decision):

    def __init__(self):
        super().__init__()

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        return createUpgradeMove(unit.id)


class GoToMine(Decision):

    def __init__(self, destination: Tuple[int, int]):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None

    def next_move(self, unit: Unit, **kwargs) -> Union[Tuple[Moves, int, Direction, int], Tuple[Moves, int]]:
        print("H")
        if self.path is None:
            self.current = (unit.x, unit.y)
            print(self.path)
            self.path = kwargs.get('bfs')(self.current, self.destination)
            self.time = len(set(self.path) - set(self.current)) + 2
            self.path.pop(0)
        if len(self.path) == 0:
            return createMineMove(unit.id)
        if self.path[0] != self.destination:
            return createDirectionMove(unit.id, unit.position(self.path.pop(0)), MAX_MOVEMENT_SPEED[Unit.WORKER])

    def reset(self):
        self.current = None
        self.path = None


def closest_pairs(start, targets):
    """

    """
    lst = []
    print(targets)
    targets = [(i[0], i[1], -1) for i in targets.keys()]
    for i, j in enumerate(start):
        lst.append((targets[i][0], targets[i][1], j[2]))
    return lst
