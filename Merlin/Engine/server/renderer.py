import pygame
class RenderingEngine:

    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)
    def __init__(self, factory, gameMap):
        self.tile_list = []
        self.map = gameMap
        self.miscellaneous = {}
        self.set_drawing_configuration()
        # find size of tiles required to be at max 720x720 
        # 720 = cells_x*tile_size,
        pygame.init()
        self.factory = factory
        self.screen = pygame.display.set_mode((920, 720))
        pygame.display.set_caption("Battlecode: Merlin.io")
        pygame.event.set_blocked(pygame.MOUSEMOTION)

    def set_drawing_configuration(self):
        self.cells_y = len(self.map.map)
        self.cells_x = len(self.map.map[0])
        self.tile_size = min(920/self.cells_x, 920/self.cells_y)
        self.tile_size_x = min(920/self.cells_x, 920/self.cells_y)
        self.tile_size_y = min(920/self.cells_x, 920/self.cells_y)
        self.grid_bias = 0

    def wait(self):
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    return
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    raise SystemExit

    def update(self, gameMap, units, miscellaneous):
        self.units = units
        self.map = gameMap
        self.miscellaneous = miscellaneous

    def cell_to_pos(self, x , y):
        x = x * self.tile_size_x + self.grid_bias
        y = y * self.tile_size_y + self.grid_bias
        return x, y

    def draw(self):
        # TODO render units
        self.screen.fill(pygame.Color("black"))
        self.draw_map()
        self.draw_units()
        # TODO render miscellaneous
        self.draw_text(self.screen, str(self.miscellaneous))
        pygame.display.update()
        pygame.event.wait()

    def draw_map(self):
        row_count = 0
        for row in self.map:
            col_count = 0
            if len(row) == 0:
                raise Exception("Illegal row found")
            for tile in row:
                img = pygame.transform.scale(
                    self.factory.get_tile_image(tile), (self.tile_size, self.tile_size)
                )
                img_rect = img.get_rect()
                pos = self.cell_to_pos(col_count, row_count)
                img_rect.x = pos[0]
                img_rect.y = pos[1]
                drawn = (img, img_rect)
                self.tile_list.append(drawn)
                col_count += 1
            row_count += 1

        for tile in self.tile_list:
            self.screen.blit(tile[0], tile[1])

    def draw_units(self):
        unit_list = []
        for unit in self.units.values():
            img = pygame.transform.scale(self.factory.get_unit_image(unit), (self.tile_size, self.tile_size)
                )
            img_rect = img.get_rect()
            img_rect = img.get_rect()
            img_rect.x = unit.x * self.tile_size
            img_rect.y = unit.y * self.tile_size
            drawn = (img, img_rect)
            unit_list.append(drawn)
        for unit in unit_list:
            self.screen.blit(unit[0], unit[1])

    def draw_text(self, surface, text, color=pygame.Color('white')):
        font = pygame.font.SysFont(None, 40)
        words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
        space = font.size(' ')[0]  # The width of a space.
        max_width, max_height = (720,720)
        pos = (300, 600)
        x, y = pos
        for line in words:
            for word in line:
                word_surface = font.render(word, 0, color)
                word_width, word_height = word_surface.get_size()
                if x + word_width >= max_width:
                    x = pos[0]  # Reset the x.
                    y += word_height  # Start on new row.
                surface.blit(word_surface, (x, y))
                x += word_width + space
            x = pos[0]  # Reset the x.
            y += word_height  # Start on new row.

class BaseRenderFactory:
    def get_tile_image(self, tile):
        raise NotImplementedError
    def get_unit_image(self, unit):
        raise NotImplementedError