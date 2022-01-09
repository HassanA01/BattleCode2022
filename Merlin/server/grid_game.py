import json
from os import X_OK
from game.constants import ATTACK_DAMAGE, BUY_TIME, MINING_TIME, UPGRADE_COSTS
from game.constants import Tiles
from server import *
from copy import deepcopy

from Engine.server.grid_game import GridGame, GridGameFactory
from Engine.server.grid_game import ClientConnection
from Engine.server.maps import Map
from .unit import *
from .tile import BaseTile
from .move import *
from game.constants import Units, Direction, MAX_ATTACK_RANGE, MINING_REWARDS, TURNS_PER_PLAYER, MAX_MOVEMENT_SPEED, MAX_LEVEL

def direction_to_coord(x,y, direction, magnitude = 1):
    if direction == Direction.DOWN:
        return (x, y + magnitude)
    if direction == Direction.UP:
        return (x, y - magnitude)
    if direction == Direction.LEFT:
        return (x - magnitude, y)
    if direction == Direction.RIGHT:
        return (x + magnitude, y)
    return None, None

def is_within_map(map,x,y):
    return 0 <= x and x < len(map[0]) and 0 <= y and y < len(map)

def is_conflicting(x,y,all_units):
    return all_units.get('{},{}'.format(x, y),None) != None

def is_straight_line(x,y,targetX,targetY):
    return bool(targetX - x != 0) != bool(targetY - y != 0)


def is_free_path(x,y,targetX,targetY,grid,allUnits):
    #assumes straight line
    dx = targetX - x 
    dy = targetY - y

    newX = x 
    newY = y

    while (dx != 0 or dy != 0):
        if dx != 0:
            newX = newX + (-1 if dx < 0 else 1)
        if dy != 0:
            newY = newY + (-1 if dy < 0 else 1)
        if repr(grid[newY][newX]) == Tiles.WALL or is_conflicting(newX,newY,allUnits):
            return False
        if dx != 0:
            dx = dx + (1 if dx < 0 else -1)
        if dy != 0:
            dy = dy + (1 if dy < 0 else -1)
    return True


def can_reach(x,y,targetX,targetY,unitType):
    if not is_straight_line(x,y,targetX,targetY):
        return False
    
    dist = abs(targetX - x) + abs(targetY - y)
    return MAX_ATTACK_RANGE[unitType] <= dist


class MerlinGridGame(GridGame):
    """
    GridGame is the currently running game, it controls all game state and updates the state each turn with tick.
    This game is won by a player capturing the other player's flag, having the most resources at the end of the game, or
    by killing all the enemy player's units.

    """
    def __init__(self, player_one_connection:ClientConnection, player_two_connection:ClientConnection, gamemap:Map):
        super().__init__(player_one_connection, player_two_connection, gamemap)
        self.turns = 0
        self.totalTurns = TURNS_PER_PLAYER*2
        self.p1_flag = {}
        self.p2_flag = {}
        self.unitFactory = UnitFactory()
        # Make team's initial units
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                tile = self.grid[row][col]
                if isinstance(tile, BaseTile):
                    if row < len(self.grid)/2:
                        unit = self.unitFactory.createUnit(Units.WORKER, col, row)
                        self.add_unit(self.p1_units, unit)
                    else:
                        unit = self.unitFactory.createUnit(Units.WORKER, col, row)
                        self.add_unit(self.p2_units, unit)
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                tile = self.grid[row][col]
                if isinstance(tile, BaseTile):
                    if row < len(self.grid)/2:
                        self.setFlag(self.p1_flag, col, row)
                    else:
                        self.setFlag(self.p2_flag, col, row)

    def setFlag(self, player_flag, x, y):
        player_flag['x'] = x
        player_flag['y'] = y

    def getEnemyFlag(self,name):
        if self.p1_conn.name == name:
            return self.p2_flag
        else:
            return self.p1_flag

    def verify_move(self, k:int, v:Move, player_state, player_resources, name, enemy_units, moved_units):
        if str(k) not in player_state:
            print('ERROR: Cannot move enemy unit: {}'.format(k))
            return False
        unit = player_state[str(k)]
        if  isinstance(unit, WorkerUnit) and (unit.is_duplicating() or unit.is_mining()):
            return False
        if unit.id in moved_units:
            return False
        #Checks if unit is currently doing something preventing them from moving
        if isinstance(unit, Unit):
            if isinstance(v, MineMove):
                if not isinstance(unit, WorkerUnit):
                    return False
                if unit.is_duplicating():
                    print('ERROR: {} cannot act while duplicating'.format(k))
                    return False

                if unit.is_mining():
                    print('ERROR: {} cannot act while mining'.format(k))
                    return False

                if repr(self.grid[unit.y][unit.x]) in [Tiles.GOLD, Tiles.COPPER, Tiles.SILVER]:
                    moved_units.add(unit.id)
                    return True
                else:
                    return False
            elif isinstance(v, BuyMove):
                if unit.unitType != Units.WORKER:
                    return False
                if v.unitType not in Units._value2member_map_:
                    return False

                unitCost = UPGRADE_COSTS[v.unitType][0]
                if (player_resources < unitCost):
                    return False
                newX, newY = direction_to_coord(unit.x,unit.y,v.direction)
                if newX == None and newY == None:
                    return False
                if is_within_map(self.grid, newX, newY) and repr(self.grid[newY][newX]) != Tiles.WALL:
                    moved_units.add(unit.id)
                    return True
                else:
                    return False
            elif isinstance(v, CaptureMove):
                if unit.unitType != Units.SCOUT:
                    return False
                newX, newY = direction_to_coord(unit.x,unit.y,v.direction)
                if newX == None and newY == None:
                    return False
                enemyFlag = self.getEnemyFlag(name)
                if enemyFlag['x'] == newX and enemyFlag['y'] == newY:
                    moved_units.add(unit.id)
                    return True
                else:
                    return False
            elif isinstance(v, DirectionMove):
                if (v.magnitude > MAX_MOVEMENT_SPEED[unit.unitType]) or v.magnitude <= 0:
                    return False
                newX, newY = direction_to_coord(unit.x,unit.y,v.direction, v.magnitude)
                if not is_within_map(self.grid,newX,newY) or not is_straight_line(unit.x, unit.y, newX, newY):
                    return False
                if is_free_path(unit.x,unit.y,newX,newY,self.grid,self.all_units):
                    moved_units.add(unit.id)
                    return True
                else:
                    return False

            elif isinstance(v, AttackMove):
                x,y = unit.pos_tuple()
                v.target = direction_to_coord(x,y ,v.direction, v.length)
                targetX, targetY = v.target
                if not is_within_map(self.grid, targetX, targetY):
                    return False
                if not self.has_unit(targetX, targetY):
                    return False
                if can_reach(unit.x,unit.y, targetX, targetY, unit.unitType):
                    moved_units.add(unit.id)
                    return True
                else:
                    return False
            elif isinstance(v, UpgradeMove):
                if unit.level >= MAX_LEVEL:
                    return False
                unitCost = UPGRADE_COSTS[v.unitType][unit.level + 1]
                if player_resources >= unitCost:
                    moved_units.add(unit.id)
                    return True
                else:
                    return False
        return False

    def get_overridden_state(self):
        misc = {**self.resources, "Player 1 Flag": self.p1_flag, "Player 2 Flag":self.p2_flag}
        return self.grid, self.all_units, misc

    def get_direction_change(self, direction):
        x = 0
        y = 0
        if direction == Direction.UP:
            y = 1
        elif direction == Direction.DOWN:
            y = -1
        elif direction == Direction.LEFT:
            x = -1
        elif direction == Direction.RIGHT:
            x = 1
        return x,y

    def get_relative_location(self, x, y, direction):

        rx, ry = self.get_direction_change(direction)

        x += rx
        y += ry
        return x, y

    def make_move(self, k, v, player_state, player_name, opponent_state):
        k = str(k)
        unit = player_state[k]
        if isinstance(v, DirectionMove):
            direction = v.direction
            magnitude = v.magnitude
            x,y = unit.x, unit.y
            old_x ,old_y = x, y
            dx,dy = direction_to_coord(0,0,direction)
            x += dx*magnitude
            y += dy*magnitude
            unit.x = x
            unit.y = y
            self.move_unit(old_x, old_y, unit)
            if unit.has_flag:
                self.setFlag(self.getEnemyFlag(player_name), x, y)
        elif isinstance(v, AttackMove):
            attacked_unit = self.get_unit(v.target[0], v.target[1])
            damage = ATTACK_DAMAGE[unit.unitType][unit.level]
            attacked_unit.health -= damage
            if attacked_unit.health <= 0:
                self.del_unit(attacked_unit.x, attacked_unit.y)
                del opponent_state[str(attacked_unit.id)]
        elif isinstance(v, UpgradeMove):
            self.resources[player_name] -= UPGRADE_COSTS[unit.unitType][unit.level]
            unit.level += 1
        elif isinstance(v, BuyMove):
            self.resources[player_name] -= UPGRADE_COSTS[v.unitType][0]
            unit.action_direction = v.direction
            unit.start_duplication(v.unitType,BUY_TIME)
        elif isinstance(v, MineMove):
            unit.mining_status = MINING_TIME
            unit.action_direction = None
        elif isinstance(v, CaptureMove):
            direction = v.direction
            x,y = unit.x, unit.y
            flag = self.getEnemyFlag(player_name)
            self.setFlag(flag, x, y)
            unit.has_flag = True

    def json_str(self):
        display = deepcopy(self.grid)
        for u in self.p1_units.values():
            display[u.y][u.x] = u.string()
        for u in self.p2_units.values():
            display[u.y][u.x] = u.string()
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
        return self.grid, self.all_units, self.resources

    def getWinner(self):
        #one side has no units or their flag has reached enemy base
        #if timeout, whoever has most resources
        p1Name = self.p1_conn.name
        p2Name = self.p2_conn.name
        if self.turns > self.totalTurns:
            #timed out
            return p1Name if self.resources[p1Name] >= self.resources[p2Name] else p2Name
        
        p1FlagTile = self.grid[self.p1_flag['y']][self.p1_flag['x']]

        p2FlagTile = self.grid[self.p2_flag['y']][self.p2_flag['x']]
        if repr(p1FlagTile) == Tiles.BASE:
            if self.p1_flag['y'] > len(self.grid) / 2:
                return p2Name
            
        if repr(p2FlagTile) == Tiles.BASE:
            if self.p2_flag['y'] < len(self.grid) / 2:
                return p1Name
        
        if len(self.p1_units.values()) == 0:
            return p2Name
        
        if len(self.p2_units.values()) == 0:
            return p1Name
        
        return None


    def tick_player(self, conn:ClientConnection, current, opponent, name:str, turns):
        #Gets a list of moves from the player
        misc = {"your_flag": {},
                "enemy_flag": {}
        }

        if name  == self.p1_conn.name:
            misc = {"your_flag":self.p1_flag,
                    "enemy_flag":self.p2_flag
            }
        else:
            misc = {"your_flag":self.p2_flag,
                    "enemy_flag":self.p1_flag
            }
        moves = conn.tick(self, current, opponent, self.resources, turns, misc)

        moved_units = set() # id to n number of moves
        #Goes through each given move and verifies it is valid. If it is execute it.
        for m in moves:
            unit_id, move_object = m
            # Key is unit id , value is move arguments
            if self.verify_move(unit_id, move_object, current, self.resources[name], name, opponent, moved_units):
                self.make_move(unit_id, move_object, current, name, opponent)

    # Creates the workers desired duplicate at a given tile.
    def create_duplicate(self,unit:WorkerUnit, location):
        return self.unitFactory.createUnit(unit.duplicating_to, location[0], location[1])


    #tick is run each turn and updates the game state
    def tick(self):
        #Gets the moves from each player and executes.
        if self.turns % 2 == 0:
            self.tick_player(self.p1_conn, self.p1_units,
                         self.p2_units, self.p1_conn.name, self.turns)
        else:
            self.tick_player(self.p2_conn, self.p2_units,
                            self.p1_units, self.p2_conn.name, self.turns)

        # manage each units actions that change game state overall.
        #Checks if any units are duplicating, if they are increment the status and create a new unit if they are complete
        for unit in list(self.all_units.values()):
            player = self.get_unit_player(unit)
            if isinstance(unit, WorkerUnit) and unit.is_duplicating():
                unit.duplication_time -= 1
                location = direction_to_coord(unit.x,unit.y,unit.action_direction)
                if unit.duplication_time <= 0 and not self.has_unit(location[0], location[1]):
                    self.add_unit(player, self.create_duplicate(unit, location))
                    unit.finish_duplicating()

        #Checks if any units are mining, if they are increment the status and add resources if they complete
        for unit in self.all_units.values():
            player_name = self.get_unit_player_name(unit)
            print(player_name, ":", unit.id, unit.unitType)
            #if unit.unitType == Units.WORKER:
                #print(unit.__dict__)
            if isinstance(unit, WorkerUnit) and unit.is_mining():
                #print("Unit mining:", unit)
                unit.mining_status -= 1
                if unit.mining_status == 0:
                    unit.mining_status = -1
                    tile = self.get_tile(unit)
                    self.resources[player_name] += MINING_REWARDS[str(tile)]
        self.turns += 1

class MerlinGridGameFactory(GridGameFactory):
    def getGame(self, connections, map)->MerlinGridGame:
        return MerlinGridGame(*connections, map)

    def serialize_unit(self, unit:GameUnit):
        d = {
            "x": unit.x,
            "y": unit.y,
            "health": unit.health,
            "level": unit.level,
            "has_flag": unit.has_flag,
            "id": unit.level,
            "unitType": unit.unitType.value
        }
        return d
    def deserialize_unit(self, unit:dict):
        unit["unitType"] = Units(unit["unitType"])
        return LoadUnit(**unit)