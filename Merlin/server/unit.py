from server import *
from Engine.server.units import Unit

from game.constants import Units

class UnitFactory():

    id_count = 0

    def createUnit(self,uid, x,y):
        '''
        Deserializes unit, this is the factory used to create new units
        '''
        unit = None
        if uid == Units.WORKER:
            unit = WorkerUnit(x,y, self.id_count)
        elif uid == Units.SCOUT:
            unit = ScoutUnit(x,y, self.id_count)
        elif uid == Units.KNIGHT:
            unit = KnightUnit(x,y, self.id_count)
        elif uid == Units.ARCHER:
            unit = ArcherUnit(x,y, self.id_count)
        else:
            raise Exception(f"Error duplicating into invalid unit uid: {uid}")
        self.id_count += 1
        return unit

class LoadUnit(Unit):
    def __init__(self,x,y,unitType:Units, id, health, level, has_flag):
        self.health = health
        self.level = level
        self.has_flag = has_flag
        super().__init__(x,y,unitType, id)
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self.unitType.value)

    def string(self):
        return str(self.unitType.value)
class GameUnit(Unit):
    def __init__(self,x,y,unitType:Units, id, health, speed):
        self.health = health
        self.speed = speed
        self.level = 1
        self.has_flag = False
        super().__init__(x,y,unitType, id)
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return NotImplemented()

    def string(self):
        return str(self.unitType.value)


class WorkerUnit(GameUnit):

    def __init__(self, x, y, id):
        super().__init__(x, y, Units.WORKER, id, 100, 1)
        self.duplication_time = -1
        self.duplicating_to = None
        self.duplicating = False
        self.action_direction = None
        self.mining_status = -1

    def __repr__(self):
        return "W"

    def is_duplicating(self):
        return self.duplicating

    def start_duplication(self, duplicating_to_type, duplication_time):
        self.duplicating = True
        self.duplicating_to = duplicating_to_type
        self.duplication_time = duplication_time

    def finish_duplicating(self):
        self.duplication_time = -1
        self.duplicating_to = None
        self.duplicating = False
    
    def is_mining(self):
        return self.mining_status >= 0

class ScoutUnit(GameUnit):

    def __init__(self, x, y, id):
        super().__init__(x, y, Units.SCOUT, id, 60, 1)

    def __repr__(self):
        return "T"

    def capture(self):
        self.has_flag = True

class KnightUnit(GameUnit):

    def __init__(self, x, y, id):
        super().__init__(x, y, Units.KNIGHT, id, 130, 2)

    def __repr__(self):
        return "K"


class ArcherUnit(GameUnit):

    def __init__(self, x, y, id):
        super().__init__(x,y, Units.ARCHER, id, 100, 1)


    def __repr__(self):
        return "A"

    def attack():
        pass

    def can_hit(direction):
        pass


