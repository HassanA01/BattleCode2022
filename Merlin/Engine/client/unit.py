from .move import Move


class Unit:
    def __init__(self, attr: dict) -> None:
        """
        Initialize a new Unit.
        """
        self.attr = attr
        self.type = attr['type']
        self.x = attr['x']
        self.y = attr['y']
        self.id = attr['id']
        self.health = attr['health']
        self.level = attr['level']

    def position(self):
        """
        Returns the current position of this Unit.
        """
        return self.x, self.y

    def direction_to(self, pos):
        """
        Returns a required direction from a unit to <pos>.
        """
        if self.y < pos[1]:
            return 'DOWN'
        elif self.y > pos[1]:
            return 'UP'
        elif self.x > pos[0]:
            return 'LEFT'
        elif self.x < pos[0]:
            return 'RIGHT'

    def move(self, *directions: (str)) -> Move:
        """
        Returns a Move for this Unit using the given <*directions>.
        """
        return Move(self.id, *directions)

    def move_towards(self, dest):
        """
        Return a Move for this Unit towards <dest>.
        """
        direction = self.direction_to(dest)
        return Move(self.id, direction)

    def nearby_enemies_by_distance(self, enemy_units: 'Units'):
        """
        Returns a sorted list of ids and their distances (in a tuple).
        """
        x = self.x
        y = self.y
        enemies = []

        for id in enemy_units.units:
            unit = enemy_units.get_unit(id)
            dist = abs(x - unit.x) + abs(y - unit.y)
            enemies.append((str(unit.id), dist))

        enemies.sort(key=lambda tup: tup[1])
        return enemies
