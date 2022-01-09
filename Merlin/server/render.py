import pygame
from Engine.server.renderer import BaseRenderFactory
from .tile import *
from .unit import *
class MerlinRenderFactory(BaseRenderFactory):
    def get_tile_image(self, tile):
        # Load images
        img = None
        if isinstance(tile, BaseTile):
            img = pygame.image.load("game/assets/base.gif")
        elif isinstance(tile, GroundTile):
            img = pygame.image.load("game/assets/empty.gif")
        elif isinstance(tile, GoldTile):
            img = pygame.image.load("game/assets/gold.gif")
        elif isinstance(tile, SilverTile):
            img = pygame.image.load("game/assets/silver.gif")
        elif isinstance(tile, CopperTile):
            img = pygame.image.load("game/assets/copper.gif")
        elif isinstance(tile, WallTile):
            img = pygame.image.load("game/assets/wall.png")
        else:
            raise Exception("Invalid tile type:", tile)
        return img

    def get_unit_image(self, unit):
        # Load images
        img = None
        id = unit.unitType
        if  id == Units.WORKER:
            img = pygame.image.load("game/assets/miner_1.webp")
        elif id == Units.SCOUT:
            img = pygame.image.load("game/assets/scout_1.png")
        elif id == Units.KNIGHT:
            img = pygame.image.load("game/assets/knight_1.png")
        elif id == Units.ARCHER:
            img = pygame.image.load("game/assets/archer_1.png")
        else:
            print(unit.id)
            raise Exception("Invalid unit type:", type(unit))
        return img