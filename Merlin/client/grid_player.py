from .helper_classes import *
from Engine.client.unit import Unit

import queue

from typing import List, Tuple, Optional


class GridPlayer:

    def __init__(self):
        self.foo = True
        self.count = 0
        self.id_dict = dict()
        self.avail_resources = dict()
        self.enemy_resources = dict()
        self.dead_tasks = {1: [], 2: [], 3: [], 4: []}         # Tasks of the dead Units
        self.queue = queue.Queue()

    def tick(self, game_map, your_units, enemy_units, resources, turns_left, your_flag, enemy_flag):
        # Task, the player codes this
        if self.count <= 1:
            half_grid = len(game_map.grid) // 2
            if your_flag['y'] > half_grid:      # Our team is bottom half of screen
                pos = 0
            else:
                pos = 1
            for i in game_map.find_all_resources():
                if (i[1] > half_grid) ^ bool(pos):
                    self.enemy_resources[i] = -1
                else:
                    self.avail_resources[i] = -1
            self.initialize_tags(your_units)
        self.count += 1
        self.initialize_decision(your_units)
        print("turn taken: ", self.count)
        for i in self.avail_resources:
            self.queue.put(GoToMine(i))
        start = [self.id_dict[i] for i in your_units.get_all_unit_of_type("WORKER") if self.id_dict[i].time <= 0]
        closest = closest_pairs(start, self.avail_resources)
        lst = []
        for i in closest:
            self.id_dict[i[2]].set_decision(GoToMine((i[0], i[1])))
            lst.append(self.id_dict[i[2]].next_move())

        return lst


    def initialize_decision(self, your_units):
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

    def closest_resources(self, unit: Unit) -> (int, int):
        """
        Returns the coordinates of the closest resource to <unit>.
        """
        c, r = unit.position()
        return min(abs(c_2 - c) + abs(r_2 - r) for (c_2, r_2) in self.avail_resources.keys()) if self.avail_resources else -1, -1

    def adding_tags(self, your_units: List[Unit]) -> Tuple[List[Unit], List[Unit]]:
        """
        takes systems ids, map them to our own id system, if it has some new ids, get units info,
        assign them new ids in our system. If its a new ID, give a custom new ID in our system as well.
        If ID is not seen on map then clear ID from hashmap ( worker has been destroyed ).

        Returns all the GameUnits that have been deleted and all GameUnits that have been created
        """
        temp_dict = set(your_units.get_all_unit_ids()) - set(self.id_dict)
        lst1 = list()               # New units (on the board)
        for i in temp_dict:
            game_unit = GameUnit(i, your_units.get_unit(i))
            self.id_dict[i] = game_unit
            lst1.append(game_unit)
        lst = list()                # Deleted units (not on the board)
        temp = set(self.id_dict) - set(your_units.get_all_unit_ids())
        for i in temp:
            lst.append(self.id_dict.pop(i))

        return lst1, lst

    def initialize_tags(self, your_units: List[Unit]) -> None:
        """
        """
        lst_unit_ids = your_units.get_all_unit_ids()

        self.id_dict = {i: GameUnit(i, your_units.get_unit(i)) for i in lst_unit_ids}

    def first_phase(self, lst_workers):
        """
        """


class Decision:

    def __init__(self):
        """"""
        self.time = 1


class GameUnit(Unit):

    def __init__(self, attr):
        super(attr).__init__()
        self.decision = queue.Queue()
        self.time = 0
        self.current = None

    def set_decision(self, decision: Decision) -> None:
        self.decision.put(decision)

    def make_decision(self) -> None:
        """
        Dequeue a decision and sets it to self.current
        """
        if self.time == 0 and not self.decision.empty():
            self.current = self.decision.get()
            self.time = self.current.time
        else:
            self.current = None

    def return_decision(self):
        # Returns helper classes functions at top or None. Annotation is incorrect.
        self.time -= 1
        if isinstance(self.current, Mine):
            return createMineMove(self.id)
        if isinstance(self.current, GoTo):
            return self.move_towards(self.current.next_move())
        if isinstance(self.current, Buy):
            return createBuyMove(self.id, self.current.piece_type, self.direction)
        if isinstance(self.current, Attack):
            return self.current.attack()
        if isinstance(self.current, Upgrade):
            pass                                # Finish this line
        if isinstance(self.current, GoToMine):
            temp = self.current.next_move()
            if temp != (-1, -1):
                return self.move_towards(self.current.next_move())
            else:
                return createMineMove(self.id)
        return None


class Attack(Decision):

    def __init__(self, where):
        super().__init__()
        self.where = where

    def attack(self):
        """
        """


class Mine(Decision):

    def init(self):
        super().__init__()
        self.time = 2


class GoTo(Decision):

    def __init__(self, destination):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None
        self.time = 0

    def next_move(self):
        """
        Tells us what the next move of this unit is.
        """
        if self.path is not None and self.path[0] != self.destination:
            return self.path.pop(0)

    def set_current_pos(self, current, game_map):
        self.current = current
        self.path = game_map.bfs(self.current, self.destination)
        self.time = len(set(self.path)-set(self.current))


class Buy(Decision):

    def __init__(self, piece_type, direction):
        super().__init__()
        self.piece_type = piece_type
        self.direction = direction


class Upgrade(Decision):

    def __init__(self):
        pass


class GoToMine(Decision):

    def __init__(self, destination):
        super().__init__()
        self.current = None
        self.destination = destination
        self.path = None
        self.time = 2

    def next_move(self):
        if self.path is not None and self.path[0] != self.destination:
            return self.path.pop(0)
        return -1, -1

    def set_current_pos(self, current, game_map):
        self.current = current
        self.path = game_map.bfs(self.current, self.destination)
        self.time += len(set(self.path)-set(self.current))


def closest_pairs(start, targets):
    """

    """
    lst = []
    start = [(i.x, i.y, i.id) for i in start]
    targets = [(i.x, i.y, -1) for i in targets]
    for i, j in enumerate(start):
        lst.append((targets[i][0], targets[i][1], j[2]))

    return lst