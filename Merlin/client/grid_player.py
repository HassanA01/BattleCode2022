#from helper_classes import *
from game.constants import *

class GridPlayer:

    def __init__(self):
        self.foo = True
        self.count = 0
        self.id_dict = dict()

    def tick(self, game_map, your_units, enemy_units, resources, turns_left, your_flag, enemy_flag):
        # Task, the player codes this
        self.count += 1
        print("turn taken: ", self.count)
        return []

    def adding_tags(self, your_units):
        """
        takes systems ids, map them to our own id system, if it has some new ids, get units info,
        assign them new ids in our system. If its a new ID, give a custom new ID in our system as well.
        If ID is not seen on map then clear ID from hashmap ( worker has been destroyed ).

        Returns all the GameUnits that have been deleted and all GameUnits that have been created
        """
        temp_dict = set(your_units.get_all_unit_ids()) - set(self.id_dict)
        lst1 = list()
        for i in temp_dict:
            gameUnit = GameUnit(i, your_units.get_unit(i))
            self.id_dict[i] = gameUnit
            lst1.append(gameUnit)
        lst = list()
        temp = set(self.id_dict) - set(your_units.get_all_unit_ids())
        for i in temp:
            lst.append(self.id_dict.pop(i))

        return lst, lst1

    def initialize_tags(self, your_units):
        """
        """
        lst_unit_ids = your_units.get_all_unit_ids()

        self.id_dict = {i: GameUnit(i, your_units.get_unit(i)) for i in lst_unit_ids}

    def first_phase(self, lst_workers):




class GameUnit:

    def __init__(self, idd, unit_type):
        self.decision = None
        self.id = idd
        self.type = unit_type

    def set_decision(self, decision):
        self.decision = decision


class Decision:

    def __init__(self):
        """"""


class Attack(Decision):

    def __init__(self, where):
        super().__init__()
        self.where = where


class Mine(Decision):
    pass


class GoTo(Decision):

    def __init__(self, current, destination, path):
        super().__init__()
        self.current = current
        self.destination = destination
        self.path = path

    def next_move(self):
        """
        Tells us what the next move of this unit is.
        """
        if self.path is not None or self.path[0] != self.destination:
            return self.path[0]