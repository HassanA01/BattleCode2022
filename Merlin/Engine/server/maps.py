from .tile import TileFactory, IllegalTileException
from copy import deepcopy
class Map:
    """
    # 2D array of tile objects
    Represents a map, methods include ability to serialize and deserialize. this is the base class.
    we expect users to build on top of this
    Nested List structure, first nests list is first row elements, second list is second row elements etc.
    Map file is expected to be a string containing the
    """

    def __init__(self, map_file, tileFactory):
        self.map_file = map_file
        self.tileFactory = tileFactory
        self.map = []
        if "\n" in map_file:
            self.deserialize_from_text()
        else:
            self.deserialize()

    def print_map(self):
        display = deepcopy(self.map)
        #Prints each row
        display = [list(map(str, r)) for r in display]
        for row in display:
            print(''.join(row))

    def deserialize(self):
        """
        deserializes the map object from the human readable .map file to its state in the game.
        :return:
        """
        if not self.map_file:
            raise Exception("Error deserializing map. Map file is none")
        # note: r by default
        with open(self.map_file) as f:
            i = 0
            for line in f.readlines():
                self.map.append([])
                for char in line.strip("\n"):
                    tile = self.tileFactory.createTile(char)
                    if not tile:
                        raise IllegalTileException(f"Error: unknown symbol: {char} found in {self.map_file}")
                    self.map[i].append(tile)
                i += 1
            f.close()
        print(f"Deserialized map: {self.map_file}")

    def deserialize_from_text(self):
        """
        deserializes the map object from the human readable .map file to its state in the game.
        :return:
        """
        loaded_text = self.map_file
        for line in loaded_text.split("\n"):
            self.map.append([])
            for char in line:
                tile = self.tileFactory.createTile(char)
                if not tile:
                    raise IllegalTileException(f"Error: unknown symbol: {char} found in {self.map_file}")
                self.map[-1].append(tile)
        print(f"Loaded from text state")

    def serialize(self):
        """
        serializes the current state of the map object into a .map human-readable text file representation.
        Essentially converting The tiles to letters or symbols in the map file.
        :return string
        """
        output = ''
        display = deepcopy(self.map)
        #Prints each row
        display = [list(map(str, r)) for r in display]
        for row in display[:-1]:
            output+=''.join(row) + "\n"
        output += ''.join(display[-1])
        return output