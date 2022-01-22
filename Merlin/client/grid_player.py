import math
from re import I

from .helper_classes import *
from Engine.client.unit import Unit
from game.constants import *
import queue
import copy
from collections import deque
import random
from typing import List, Tuple, Type, Union, Optional, Dict

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
        self.buy = []
        self.guard = []
        self.protect = dict()

    def tick(self, game_map: Map, your_units: Units, enemy_units: Units, resources: int, turns_left: int,
             your_flag: dict,
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
            # self.initialize_tags(your_units)  # take cares of new and deleted units
            self.initialize_locked(game_map)  # which positions are always locked
            pos = 1
            if (your_flag['y'] > half_grid):
                pos = -1
            c = your_flag['y']
            r = your_flag['x']
            self.guard.append(StandbyGaurd((r + 1, c), 4))
            self.guard.append(StandbyGaurd((r - 1, c), 4))
            self.guard.append(StandbyGaurd((r, c + pos), 4))
            c = enemy_flag["y"]
            r = enemy_flag["x"]
            self.guard.append(StandbyGaurd((r, c), 4))
            self.guard.append(StandbyGaurd((r, c), 4))
            self.guard.append(StandbyGaurd((r, c), 4))
            self.guard.append(StandbyGaurd((r, c), 4))
            self.guard.append(StandbyGaurd((r, c), 4))

            for i in self.enemy_resources:
                self.guard.append(StandbyGaurd(i, 2))

            self.guard.extend([StandbyGaurd(i, 2) for i in get_prime_coordinates(game_map, self.locked, (your_flag["x"], your_flag["y"]), self.avail_resources)])
            # self.guard.append(StandbyGaurd((r, c), 2))
            # self.guard.append(StandbyGaurd((your_flag['x'], your_flag['y']),2))
            for i in range(35):
                self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.WORKER))
            self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.WORKER))
            self.buy.append(Buy(Units.KNIGHT))

            # self.buy.append(Buy(Units.KNIGHT))
            # self.buy.append(Buy(Units.KNIGHT))
            # # self.buy.append(Buy(Units.KNIGHT))
            # # self.buy.append(Buy(Units.KNIGHT))
            # # self.buy.append(Buy(Units.KNIGHT))
            # # self.buy.append(Buy(Units.KNIGHT))
            self.buy.append(Buy(Units.WORKER))
            self.buy.append(Buy(Units.WORKER))
            self.buy.append(Buy(Units.WORKER))
            self.buy.append(Buy(Units.WORKER))
            # self.buy.append(Buy(Units.WORKER))
            # self.buy.append(Buy(Units.WORKER))
            # self.buy.append(Buy(Units.SCOUT))
        print("turn taken: ", self.count)
        add2 = {Units.WORKER: [], Units.ARCHER: [], Units.KNIGHT: [], Units.SCOUT: []}
        add, delete = self.adding_tags(your_units)
        for i in delete:
            self.buy.append(Buy(i.type))
        for i in add:
            self.protect[i.get_id()] = []
            add2[your_units.get_unit(i.get_id()).type].append(i)
        locked = self.update_locked(enemy_units)
        # locked = copy.deepcopy(self.locked)
        self.initialize_decision(your_units)
        # for i in add2[Units.SCOUT]:
        #     i.add_decision(Scouting(False))
        for i in add2[Units.WORKER]:
            print(i)

            if self.buy and i.direction(your_units.get_unit(i.get_id()), locked):
                i.add_decision(self.buy.pop())
            i.add_decision(GoToMine())
        for i in add2[Units.KNIGHT]:
            if self.guard:
                i.add_decision(self.guard.pop(0))
            else:
                i.add_decision(Attack())
        for j in self.id_dict.values():
            if j.available(your_units.get_unit(j.get_id())) and self.buy and your_units.get_unit(
                    j.get_id()).type == Units.WORKER:
                j.add_decision(self.buy.pop())
            if j.available(your_units.get_unit(j.get_id())) and not self.buy and your_units.get_unit(
                    j.get_id()).type == Units.WORKER:
                c, r = your_units.get_unit(j.get_id()).position()
                print(self.protect[j.get_id()])
                if len(self.protect[j.get_id()]) < 4:
                    for x, y in ((c, r + 2), (c, r - 2), (c + 2, r), (c - 2, r)):
                        if is_within_map(locked, x, y) and (x, y) not in self.protect[j.get_id()]:
                            #self.guard.append(StandbyGaurd((x, y), 2))
                            #j.add_decision(Buy(Units.KNIGHT))
                            self.protect[j.get_id()].append((x, y))
                            break

        lst = []
        for i in self.id_dict.values():
            k = your_units.get_unit(i.id).position()
            t = i.make_decision(your_units.get_unit(i.id), game_map=game_map, avail_resources=self.avail_resources,
                                locked=locked, enemy_units=enemy_units, your_units=your_units)
            print(i.id, your_units.get_unit(i.id).type, t)
            locked[k[1]][k[0]] = 1
            # if t[0] in (Moves.ATTACK, Moves.BUY, Moves.MINE, Moves.CAPTURE):
            #     locked[k[1]][k[0]] = 1
            # print(t[0] == Moves.DIRECTION)
            if t is not None and t[0] == Moves.DIRECTION:
                coord = coordinate_from_direction(k[0], k[1], t[2])
                # locked[k[1]][k[0]] = 0
                locked[coord[1]][coord[0]] = 1

            if t is None and your_units.get_unit(i.id).type == Units.KNIGHT:
                t = createUpgradeMove(i.id)
            lst.append(t)
        # print(your_units.units)
        # for i in locked:
        # print(''.join(str(e) for e in i))
        self.count += 1
        lst = list(filter((None).__ne__, lst))
        # print(lst)

        return lst

    def initialize_locked(self, game_map: Map) -> None:
        """
        Copies the game map to a list.
        Elements of list are 1 if position is blocked and 0 otherwise.
        """
        for i in game_map.grid:
            self.locked.append([1 if j == 'X' else 0 for j in i])

    def update_locked(self, enemy_units: Units) -> List[List[int]]:
        """
        Updates the map with the enemy positions and sets positions as 1 (blocked) and returns the updated list.
        """
        locked = copy.deepcopy(self.locked)
        for i in enemy_units.get_all_unit_ids():
            unit = enemy_units.get_unit(i)
            locked[unit.y][unit.x] = 1
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

    def adding_tags(self, your_units: Units) -> Tuple[List[Unit], List[Unit]]:
        """
        takes systems ids, map them to our own id system, if it has some new ids, get units info,
        assign them new ids in our system. If its a new ID, give a custom new ID in our system as well.
        If ID is not seen on mp then clear ID from hashmap ( worker has been destroyed ).
        Returns all the GameUnits that have been deleted and all GameUnits that have been created
        """
        temp_dict = set(your_units.get_all_unit_ids()) - set(self.id_dict.keys())
        lst1 = list()  # New units (on the board)
        for i in temp_dict:
            game_unit = GameUnit(i, your_units.get_unit(i).type)
            self.id_dict[i] = game_unit

            lst1.append(game_unit)
        lst = list()  # Deleted units (not on the board)
        temp = set(self.id_dict) - set(your_units.get_all_unit_ids())
        for i in temp:
            lst.append(self.id_dict.pop(i))
            if i.type == Units.WORKER:
                self.avail_resources[i.id] = -1
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

    def __init__(self, idd: int, type):
        self.type = type
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
            # self.time = self.current.time

    def make_decision(self, unit: Unit, **kwargs) -> None:
        """
        Dequeues a decision and returns the corresponding move
        """
        print('decision', self.decision)
        if self.current is None:
            return
        if self.current.are_u_done(unit, **kwargs):
            self.current = self.decision.get(unit.type)
        return self.current.next_move(unit, **kwargs)

    def direction(self, unit: Unit, locked: List[List[int]]) -> Direction:  # TODO #2 need to implement properly
        """
        Returns an empty location direction adjacent to Unit's position.
        """
        c = unit.x
        r = unit.y
        dir = None
        for i in ((c, r + 1), (c, r - 1), (c + 1, r), (c - 1, r)):
            if is_within_map(locked, i[0], i[1]) and locked[i[1]][i[0]] != 1:
                dir = direction_to(unit, i)
                break
        return dir

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
    """
    currently just a simple program that attacks the nearest enemy unit. Otherwise does nothing.
    """

    def __init__(self, where: Tuple[int, int] = None):
        super().__init__()
        self.where = where
        self.pref = None

    def are_u_done(self, unit, **kwargs):
        return False

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        enemy_units = kwargs.get('enemy_units')
        locked = kwargs.get('locked')
        k = unit.position()
        # kwargs.get('locked')[k[1]][k[0]] = 1
        enemy_ids = enemy_units.get_all_unit_ids()
        r, c = unit.position()
        if len(enemy_ids) == 0:
            if self.dest is None:
                return createDirectionMove(unit.id, get_random_direction(), 1)
            elif (r, c) == self.dest:
                self.dest = None
                return createDirectionMove(unit.id, get_random_direction(), 1)
            else:
                distance = bfs(locked, (r, c), self.dest)
                if distance is not None and unit.type == UNITS.KNIGHT:
                    if len(distance) > 3 and direction_to(unit, distance[1]) == direction_to(unit, distance[2]):
                        return createDirectionMove(unit.id, direction_to(unit, distance[1]), 2)
                    return createDirectionMove(unit.id, direction_to(unit, distance[1]), 1)

        closest = 1000
        position = unit.position()
        self.dest = None
        if unit.type == Units.KNIGHT:
            if self.pref is not None:
                if self.pref in enemy_ids:
                    x, y = enemy_units.get_unit(self.pref).position()
                    if abs(x - r) + abs(y - c) == 1:
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), 1)

                    else:
                        position = enemy_units.get_unit(self.pref).position()
                else:
                    self.pref = None
            else:
                for enemy_id in enemy_ids:
                    x, y = enemy_units.get_unit(enemy_id).position()
                    if abs(x - r) + abs(y - c) == 1:
                        self.pref = enemy_id
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), 1)

                    if self.pref is None:
                        distance = bfs(locked, (r, c), (x, y))
                        if distance is not None:
                            if len(distance) < closest:
                                closest = len(distance)
                                position = (x, y)

            distance = bfs(locked, (r, c), position)
            self.dest = position
            if distance is not None:
                if len(distance) > 4 and direction_to(unit, distance[1]) == direction_to(unit, distance[2]):
                    return createDirectionMove(unit.id, direction_to(unit, distance[1]), 2)
                return createDirectionMove(unit.id, direction_to(unit, distance[1]), 1)
            else:
                return createDirectionMove(unit.id, get_random_direction(), 1)

        elif unit.type == Units.ARCHER:
            for enemy_id in enemy_ids:
                x, y = enemy_units.get(enemy_id).pos_tuple
                if abs(x - r) + abs(y - c) == 1 and enemy_units.get_unit(enemy_id).type == UNITS.KNIGHT:
                    if is_within_map(locked, r - 1, c) and locked[r - 1][c] == 0:
                        return createDirectionMove(unit.id, Direction.DOWN, 1)
                    elif is_within_map(r, c - 1) and locked[r][c - 1] == 0:
                        return createDirectionMove(unit.id, Direction.LEFT, 1)
                    elif is_within_map(r, c + 1) and locked[r][c + 1] == 0:
                        return createDirectionMove(unit.id, Direction.RIGHT, 1)
                    elif is_within_map(r + 1, c) and locked[r + 1][c] == 0:
                        return createDirectionMove(unit.id, Direction.UP, 1)

            if self.pref is not None:
                if self.pref in enemy_ids:
                    x, y = enemy_units.get_unit(self.pref).position()
                    if abs(x - r) <= 2 and y == c:
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), abs(x - r))
                    elif x == r and abs(y - c) <= 2:
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), abs(x - r))
                    else:
                        position = (x, y)
                else:
                    self.pref = None
            else:
                for enemy_id in enemy_ids:
                    x, y = enemy_units.get(enemy_id).pos_tuple
                    if abs(x - r) <= 2 and y == c:
                        self.pref = enemy_id
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), abs(x - r))
                    elif x == r and abs(y - c) <= 2:
                        self.pref = enemy_id
                        self.dest = (x, y)
                        return createAttackMove(unit.id, direction_to(unit, (x, y)), abs(x - r))

                    distance = bfs(locked, (r, c), (x, y))
                    if distance is not None and len(distance) < closest:
                        closest = len(distance)
                        position = (x, y)

            distance = bfs(locked, (r, c), position)
            if distance is not None:
                return createDirectionMove(unit.id, direction_to(unit, distance[1]), 1)
            return createDirectionMove(unit.id, get_random_direction(), 1)

    def __str__(self) -> str:
        return 'Attack'


class Mine(Decision):

    def __init__(self):
        super().__init__()
        self.time = 2
        self.mined = False

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int]:
        self.time -= 1
        if self.mined is False:
            self.mined = True
            return createMineMove(unit.id)

    def __str__(self) -> str:
        return 'Mine'

    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return self.time <= 0


def get_enemy_in_block(unit: Unit, enemy: Units, block_size: int, location: Tuple[int, int], locked) -> List[
    Tuple[int, int]]:
    enemy_units = enemy
    enemy_ids = enemy_units.get_all_unit_ids()
    closest = 1000
    position = unit.position()
    r, c = unit.position()
    for enemy_id in enemy_ids:
        x, y = enemy_units.get_unit(enemy_id).position()
        if abs(x - location[0]) <= block_size and abs(y - location[1]) <= block_size:
            if abs(x - r) + abs(y - c) == 1:
                return createAttackMove(unit.id, direction_to(unit, (x, y)), 1)
            distance = bfs(locked, (r, c), (x, y))
            if distance is not None:
                if len(distance) < closest:
                    closest = len(distance)
                    position = (x, y)
    if position == (r, c):
        path = bfs(locked, (r, c), location)
    else:
        path = bfs(locked, (r, c), position)
    if path is not None:
        return createDirectionMove(unit.id, direction_to(unit, path[1]), 1)


class StandbyGaurd(Decision):
    def __init__(self, dest: Tuple[int, int], size: int):
        super().__init__()
        self.dest = dest
        self.size = size

    def next_move(self, unit: Unit, **kwargs):
        enemy_units = kwargs.get('enemy_units')
        # kwargs.get('locked')[k[1]][k[0]] =
        locked = kwargs.get('locked')
        enemy_ids = enemy_units.get_all_unit_ids()
        closest = 1000
        position = unit.position()
        r, c = unit.position()
        for enemy_id in enemy_ids:
            x, y = enemy_units.get_unit(enemy_id).position()
            if abs(x - self.dest[0]) <= self.size and abs(y - self.dest[1]) <= self.size:
                if abs(x - r) + abs(y - c) == 1:
                    return createAttackMove(unit.id, direction_to(unit, (x, y)), 1)
                distance = bfs(locked, (r, c), (x, y))
                if distance is not None:
                    if len(distance) < closest:
                        closest = len(distance)
                        position = (x, y)
        if position == (r, c):
            path = bfs(locked, (r, c), self.dest)
        else:
            path = bfs(locked, (r, c), position)
        if path is not None:
            return createDirectionMove(unit.id, direction_to(unit, path[1]), 1)

    def are_u_done(self, unit: Unit, **kwargs):
        return False


class GoTo(Decision):

    def __init__(self, destination: Tuple[int, int]):
        super().__init__()
        self.destination = destination

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Moves, int, Direction, int]:
        k = unit.position()
        # kwargs.get('locked')[k[1]][k[0]] = 1
        if k != self.destination:
            # bfs1 = kwargs.get('game_map')
            path = bfs(kwargs.get('locked'), k, self.destination)
            if path is None:
                return
            next_pos = path[1]
            # kwargs.get('locked')[next_pos[1]][next_pos[0]] = 1
            return createDirectionMove(unit.id, direction_to(unit, next_pos), MAX_MOVEMENT_SPEED[Units.WORKER])

    def reset(self):
        self.current = None
        self.path = None

    def are_u_done(self, unit: Unit, **kwargs):
        return unit.position() == self.destination

    def __str__(self) -> str:
        return 'GoTo'


class Buy(Decision):

    def __init__(self, piece_type: Units):
        super().__init__()
        self.piece_type = piece_type
        self.bought = False
        self.time = 2

    def next_move(self, unit: Unit, **kwargs) -> Optional[Tuple[Moves, int, Type, Direction]]:
        self.time -= 1
        locked = kwargs.get('locked')
        c = unit.x
        r = unit.y
        dir = None
        t = set(i.position() for i in kwargs.get('your_units').units.values())
        t = t.union(set(i.position() for i in kwargs.get('enemy_units').units.values()))
        for i in {(c, r + 1), (c, r - 1), (c + 1, r), (c - 1, r)} - t:
            if is_within_map(locked, i[0], i[1]) and locked[i[1]][i[0]] != 1:
                dir = direction_to(unit, i)
                print(i, dir)
                break
        if dir is not None and self.bought is False:
            self.bought = True
            return createBuyMove(unit.id, self.piece_type, dir)

    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return self.time <= 0

    def __str__(self) -> str:
        return 'Buy'


# class StayAndGuard(Decision):
#     def __init__(self):


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
        self.destination = destination
        self.mined = False
        self.time = 2
        self.lvl = 0

    def next_move(self, unit: Unit, **kwargs) -> Union[Tuple[Moves, int, Direction, int], Tuple[Moves, int]]:
        k = unit.position()
        if self.destination is None:
            k = kwargs.get('game_map')
            self.destination = closest_resource(kwargs.get('avail_resources'), unit)
            if k.is_tile_type(self.destination[0], self.destination[1], Tiles.SILVER):
                self.lvl = 1
            if k.is_tile_type(self.destination[0], self.destination[1], Tiles.GOLD):
                self.lvl = 2
            kwargs.get('avail_resources')[self.destination] = unit.id
        if self.lvl > 0:
            self.lvl -= 1
            return createUpgradeMove(unit.id)
        # kwargs.get('locked')[k[1]][k[0]] = 1

        if (unit.x, unit.y) != self.destination:
            # bfs1 = kwargs.get('game_map')
            path = bfs(kwargs.get('locked'), (unit.x, unit.y), self.destination)
            next_pos = path[1]
            #  kwargs.get('locked')[next_pos[1]][next_pos[0]] = 1
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
        self.k = []

    def get(self, unit_type: Type = None) -> Decision:
        """
        Returns a Mine decision if id is given otherwise pops the latest decision in <self.queue> and returns it.
        """
        if self.queue.empty() and unit_type == Units.WORKER:
            return Mine()
        if self.queue.empty() and unit_type == Units.KNIGHT:
            return Attack()
        self.k.pop(0)
        return self.queue.get()

    def put(self, item: Decision) -> None:
        """
        Adds a decision to the <self.queue>.
        """
        self.k.append(item)
        self.queue.put(item)

    def empty(self, type=None) -> bool:
        """
        Returns False if <type> given is Worker otherwise returns if <self.queue> is empty.
        """
        if type == Units.WORKER:
            return False
        if type == Units.KNIGHT:
            return False
        return self.queue.empty()

    def __str__(self) -> str:
        return str(self.k)


class Scouting(Decision):

    def __init__(self, moving):
        super().__init__()
        self.moving = moving
        self.path = []

    def are_u_done(self, unit: Unit, **kwargs) -> bool:
        return False

    def next_move(self, unit: Unit, **kwargs) -> Tuple[Move, int]:
        """

        :param unit:
        :param Kwargs:
        :return:
        """
        if not self.moving:
            self.path = scout_path(kwargs.get('game_map'), kwargs.get('locked'), unit.position())
            self.moving = True

        next_pos = self.path[1]
        #  kwargs.get('locked')[next_pos[1]][next_pos[0]] = 1
        # print("next Position: ", next_pos)
        # print("Path to next pos: ", self.path)
        return createDirectionMove(unit.id, direction_to(unit, next_pos), MAX_MOVEMENT_SPEED[Units.WORKER])


def scout_path(game_map: Map, locked: List[List[int]], pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Computes a lurker path for scout to basically run around the map getting info, attracting and distracting
    enemy Units.

    :param game_map: a matrix of coordinates of the game map
    :param locked: a matrix of coordinates of game map with locked coordinates
    :param pos: position of Scout unit.
    :return: A path the scout will travel
    """

    path = []
    resources = game_map.find_all_resources()
    bottom_right_resources = [resource for resource in resources if resource[0] >= len(locked[0]) // 2 and resource[1] < len(locked) // 2]
    top_right_resources = [resource for resource in resources if resource[0] >= len(locked[0]) // 2 and resource[1] >= len(locked) // 2]
    bottom_left_resources = [resource for resource in resources if resource[0] < len(locked[0]) // 2 and resource[1] < len(locked) // 2]
    top_left_resources = [resource for resource in resources if resource[0] < len(locked[0]) // 2 and resource[1] >= len(locked) // 2]

    enemy_flag_pos = closest_flag(game_map, pos, Tiles.BASE)

    if enemy_flag_pos[1] >= len(locked) // 2:
        path.extend(create_semi_path(copy.deepcopy(bottom_right_resources), pos, locked))
        path.extend(create_semi_path(copy.deepcopy(top_right_resources), pos, locked))
        curr_pos = path[-1]
        enemy_flag_pos = closest_flag(game_map, curr_pos, Tiles.BASE)
        path_to_enemy_flag = bfs(locked, curr_pos, enemy_flag_pos)
        path.extend(path_to_enemy_flag)
        path.extend(create_semi_path(copy.deepcopy(top_left_resources), path[-1], locked))
        path.extend(create_semi_path(copy.deepcopy(bottom_left_resources), path[-1], locked))

    else:
        path.extend(create_semi_path(copy.deepcopy(top_right_resources), pos, locked))
        path.extend(create_semi_path(copy.deepcopy(bottom_right_resources), pos, locked))

    # path.extend(create_semi_path(copy.deepcopy(bottom_right_resources), pos, locked))
    # path.extend(create_semi_path(copy.deepcopy(top_right_resources), pos, locked))
    #
    #
    # print("resources: ", resources)
    # print("scout_path", path)
    # curr_pos = path[0]
    # enemy_flag_pos = closest_flag(game_map, curr_pos, Tiles.BASE)
    # path_to_enemy_flag = bfs(locked, curr_pos, enemy_flag_pos)
    #
    # if path_to_enemy_flag is not None:
    #     curr_pos = path_to_enemy_flag[-1]
    #     path.extend(path_to_enemy_flag)
    #
    # path.extend(create_semi_path(copy.deepcopy(top_left_resources), pos, locked))
    # path.extend(create_semi_path(copy.deepcopy(bottom_left_resources), pos, locked))

    # path.extend(create_semi_path(left_resources, curr_pos, locked))

    return path


def create_semi_path(resources: List[Tuple[int, int]],
                     position: Tuple[int, int], locked: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Creates part of the full path for Scout
    :param locked: Grid with locked positions
    :param resources: List of resources to visit
    :param position: Position of Scout
    :return: Half of the Scouts full path
    """
    index = 0
    path = []

    while index < len(resources) - 1:
        print("path,", path)
        if not path:
            curr_pos = position
        else:
            curr_pos = path[-1]
        print("right resources:", resources)
        closest = min_distance(curr_pos, resources)
        print("curr_pos: ", curr_pos, " closest resource: ", closest, "\n")
        temp_path = bfs(locked, curr_pos, closest)
        random_stop_point = random.randint(2, 4)
        resources.remove(closest)
        if temp_path is not None and len(temp_path) > 2:
            path.extend([i for i in temp_path[:-random_stop_point] if i not in path])

        elif temp_path is not None:
            path.extend(temp_path[:-1])

    return path


def closest_flag(game_map: Map, position: Tuple[int, int], tileType: Tiles) -> (int, int):
    """
    Returns the coordinates of the closest tile to <unit>.
    """
    locations = game_map.find_all_tiles_of_type(tileType)
    c, r = position
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


def min_distance(pos: Tuple[int, int], resources: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Gets the closest resource from the units position and returns the position

    :param pos: Position of Unit that will move
    :param resources: List of all the resources on the map
    :return: Returns the position of the closest resource.
    """

    closest = 999999
    result = None
    for pos2 in resources:
        distance = abs(pos2[0] - pos[0]) + abs(pos2[1] - pos[1])
        if distance < closest:
            result = pos2
            closest = distance
    return result


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
    if start == dest or \
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
            if is_within_map(graph, adj[0], adj[1]) and adj not in vis and (graph[adj[1]][adj[0]] != 1 or adj == dest):
                queue.append(path + [adj])
                vis.add(adj)


def is_within_map(map: List[List[int]], x: int, y: int) -> bool:
    """
    Returns if the coordinate (<x>, <y>) is in <map>.
    """

    return 0 <= x < len(map[0]) and 0 <= y < len(map)

# def bipartite_graph_min_weight(source: List[Tuple[int, int, int]], target: List[Tuple[int, int]]) -> Dict[int:
# Tuple[int, int]]: all_combinations = combinations(target, len(source)) temp = [] for target_combo in
# all_combinations: lst = [] for index, source_combo in enumerate(source): dist = abs(source_combo[1] - target_combo[
# index][1]) + abs(source_combo[0] - target_combo[index][0]) lst.append((source_combo[2], dist, target_combo[index]))
# (id, distance, target) temp.append(lst) min_combo = min(temp, key=lambda x: sum(i[1] for i in x)) min_dict = {i[0]:
# i[2] for i in min_combo} return min_dict


def get_prime_coordinates(game_map: Map, locked: List[List[int]], your_flag: Tuple[int, int], avail_resources: dict) -> List[Tuple[int, int]]:
    """

    :param avail_resources:
    :param game_map: game map
    :param locked: locked positions grid
    :param your_flag: coordinate of our flag
    :return: list of locations for knights to defend
    """
    coordinates = []
    for resource_coord in avail_resources.keys():
        for i in range(resource_coord[0] - 2, resource_coord[0] + 3):
            for j in range(resource_coord[1] - 2, resource_coord[1] + 3):
                if is_within_map(locked, i, j):
                    if locked[j][i] != 1:
                        if (i, j) not in avail_resources.keys():
                            if abs(i - resource_coord[0]) + abs(j - resource_coord[1]) == 3:
                                coordinates.append((i, j))
                            elif abs(i - resource_coord[0]) + abs(j - resource_coord[1]) == 2:
                                if abs(i - resource_coord[0]) == 2 and j == resource_coord[1]:
                                    coordinates.append((i, j))
    random.shuffle(coordinates)

    return coordinates

