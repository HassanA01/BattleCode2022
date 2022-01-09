#Figure out import

from Engine.server.move import Move

from game.constants import Moves, Direction, Units
class MerlinMoveFactory:
    def createMove(self, moveid:int, restInfo):
        '''
        Deserializes move, this is the factory used in client_connection.py 
        '''
        valids = []
        for move in Moves:
          valids.append(move)
          if move.value == moveid:
            moveid = move
        if not moveid in valids:
          raise Exception(f"Invalid move id: {moveid}")
        restInfo = restInfo[1:]
        if moveid == Moves.ATTACK:
          if len(restInfo) != 3:
            raise Exception()
          
          direction = restInfo[1]
          magnitude = restInfo[2]
          if direction not in Direction._value2member_map_ or magnitude <= 0:
            raise Exception
          
          return AttackMove(*restInfo)
        elif moveid == Moves.BUY:
          if len(restInfo) != 3:
            raise Exception(f"Illegal size of action, requires 2: {restInfo}")
  
          unitType = restInfo[1]
          direction = restInfo[2]

          if unitType not in Units._value2member_map_ or direction not in Direction._value2member_map_:
            raise Exception

          return BuyMove(*restInfo)
        elif moveid == Moves.CAPTURE:
          if len(restInfo) != 2:
            raise Exception(f"Illegal size of action, requires 2: {restInfo}")
          
          direction = restInfo[1]

          if direction not in Direction._value2member_map_:
            raise Exception

          return CaptureMove(*restInfo)
        elif moveid == Moves.DIRECTION:
          if len(restInfo) != 3:
            raise Exception(f"Illegal size of action, requires 3: {restInfo}")
          direction = restInfo[1]
          magnitude = restInfo[2]
          if direction not in Direction._value2member_map_ or magnitude <= 0:
            raise Exception
          return DirectionMove(*restInfo)
        elif moveid == Moves.MINE:
          if len(restInfo) != 1:
            raise Exception(f"Illegal size of action, requires 1: {restInfo}")

          return MineMove(*restInfo)
        elif moveid == Moves.UPGRADE:
          if len(restInfo) != 1:
            raise Exception

          return UpgradeMove(*restInfo)
        raise Exception("Invalid move id")

def direction_to_enum(dirStr):
  for dirs in Direction:
    if dirs.value == dirStr:
      return dirs
  raise Exception(f"Invalid direction id: {dirStr}")

def unit_to_enum(unitStr):
  for unit in Units:
    if unit.value == unitStr:
      return unit
  raise Exception(f"Invalid unit id: {unit}")

class GameMove(Move):
  def __init__(self, unit):
    super().__init__(unit)

  def __repr__(self):
      '''
      Returns the serialized form of this move
      The subclass should implement this
      '''
      pass


class AttackMove(GameMove):
  def __init__(self, unit, direction, length):
      super().__init__(unit)
      self.direction = direction_to_enum(direction)
      self.length = length
  
  def __repr__(self):
      '''
      Returns the serialized form of this move
      '''
      return str({'command': Moves.ATTACK, "unit": self.unit, "target": self.target })


class UpgradeMove(GameMove):
  def __init__(self, unit):
      super().__init__(unit)
  
  def __repr__(self):
      '''
      Returns the serialized form of this move
      '''
      return str({'command': Moves.UPGRADE, "unit": self.unit})
class DirectionMove(GameMove):
  def __init__(self, unit, direction, magnitude):
      super().__init__(unit)
      self.direction = direction_to_enum(direction)
      self.magnitude = magnitude
  
  def __repr__(self):
      '''
      Returns the serialized form of this move
      '''
      return str({'command': Moves.DIRECTION, "unit": self.unit, "direction": self.direction})

class MineMove(GameMove):
    def __init__(self, unit):
        super().__init__(unit)
    
    def __repr__(self):
        '''
        Returns the serialized form of this move
        '''
        return str({'command': Moves.MINE, "unit": self.unit })


class BuyMove(GameMove):
  def __init__(self, unit, unitType, direction):
      super().__init__(unit)
      self.unitType = unit_to_enum(unitType)
      self.direction = direction_to_enum(direction)
  
  def __repr__(self):
      '''
      Returns the serialized form of this move
      '''
      return str({'command': Moves.BUY, "unitType": self.unitType, 'direction': self.direction})


class CaptureMove(GameMove):
  def __init__(self, unit, direction):
      super().__init__(unit)
      self.direction = direction_to_enum(direction)
  
  def __repr__(self):
      '''
      Returns the serialized form of this move
      '''
      return str({'command': Moves.CAPTURE, "unit": self.unit, 'direction': self.direction})
