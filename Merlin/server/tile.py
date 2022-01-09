from Engine.server.maps import TileFactory
from Engine.server.tile import Tile
from game.constants import Tiles

class MerlinTileFactory(TileFactory):

    def createTile(self, tileId):
        if tileId == Tiles.WALL:
            return WallTile()
        elif tileId == Tiles.GROUND:
            return GroundTile()
        elif tileId == Tiles.GOLD:
            return GoldTile()
        elif tileId == Tiles.SILVER:
            return SilverTile()
        elif tileId == Tiles.COPPER:
            return CopperTile()
        elif tileId == Tiles.BASE:
            return BaseTile()
        else:
            return None

class WallTile(Tile):
    def string(self):
        return f'"{Tiles.WALL}"'

    def __repr__(self):
        return Tiles.WALL

class GroundTile(Tile):
    def string(self):
        return f'"{Tiles.GROUND}"'

    def __repr__(self):
        return Tiles.GROUND

class GoldTile(Tile):
    def string(self):
        return f'"{Tiles.GOLD}"'

    def __repr__(self):
        return Tiles.GOLD

class SilverTile(Tile):
    def string(self):
        return f'"{Tiles.SILVER}"'

    def __repr__(self):
        return Tiles.SILVER

class CopperTile(Tile):
    def string(self):
        return f'"{Tiles.COPPER}"'

    def __repr__(self):
        return Tiles.COPPER

class BaseTile(Tile):
    def string(self):
        return f'"{Tiles.BASE}"'

    def __repr__(self):
        return Tiles.BASE
