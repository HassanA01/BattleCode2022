
import argparse
import socket
import os
import random
import json
from copy import deepcopy

from sys import platform
from Engine.server.renderer import RenderingEngine
from Engine.server import renderer

from server import *
from .grid_game import GridGame, GridGameFactory
from .client_connection import ClientConnection
from .maps import TileFactory, Map

class GameEngine:
    def __init__(self, renderFactory, tileFactory:TileFactory, moveFactory, gameFactory:GridGameFactory):
        self.renderFactory = renderFactory
        self.tileFactory = tileFactory
        self.moveFactory = moveFactory
        self.gameFactory = gameFactory
        self.sock = None
        self.host = None
        self.totalPlayers = 0
        self.NUMPLAYERS = 2
        self.connections = []
        self.addresses = []
        self.verbose = False
        self.render = False

    def __launchServer(self, port, timeout=8):
        ''' Launches the server on the desired port, handling OS descrepencies.
        '''
        #Creates and connects the socket connection to the clients
        sock = socket.socket()
        host = None
        
        print("Running server on platform: ", platform)
        if platform == "linux" or platform == "linux2":
            host = socket.gethostbyname(socket.gethostname())
        elif platform == "darwin":
            host = 'localhost'
        elif 'win' in platform:
            host = 'localhost'
        else:
            print(f"The platform {platform} is not natively suppported and there may be errors")
        print("Hosting server on port: ", port ," at: ", host)
        sock.bind((host,port))

        sock.settimeout(timeout)
        sock.listen(self.NUMPLAYERS)
        self.sock = sock

    def __connectNextPlayer(self):
        player = self.totalPlayers + 1
        print(f'Waiting for client {player}...')
        conn, addr = self.sock.accept()
        self.connections.append(ClientConnection(conn, f'p{player}', verbose=self.verbose, moveFactory=self.moveFactory))
        self.addresses.append(addr)
        if not self.verbose:
            conn.settimeout(10)
        print(f'Connected to client {player} at addr')
        self.totalPlayers += 1

    def __loadMap(self, file_name=None):
        #Picks a random map from the maps folder and creates the game
        if not file_name:
            file_name = 'maps/{}'.format(random.choice(os.listdir('maps')))
        print("Loading Map:", file_name)
        self.map = Map(file_name, self.tileFactory)

    def __save_state(self, game):
        game.get_overridden_state()
        current_map, current_units, current_misc = game.get_overridden_state()
        turn_units = {}
        for key, unit in current_units.items():
            turn_units[key] = self.gameFactory.serialize_unit(unit)
        turn_state = {"all_units": turn_units, "misc": current_misc}
        self.log["state"].append(turn_state)
        
    def __runGameLoop(self, gamemap):
        game = self.gameFactory.getGame(self.connections, gamemap)
        #Ticks the game unit there is a winner or the max_turns is reached
        turn = 0
        print('Game starting...')
        if self.does_render:
            current_map, current_units, current_misc = game.get_overridden_state()
            self.renderEngine.update(current_map, current_units, current_misc)
            self.renderEngine.draw()
        if self.replaysavepath:
            self.__save_state(game)
        while game.getWinner() == None:
            print("Turn:", turn)
            print()
            game.tick()
            current_map, current_units, current_misc = game.get_overridden_state()
            if self.does_render:
                self.renderEngine.update(current_map, current_units, current_misc)
                self.renderEngine.draw()
                print("Press enter on the rendered window to continue. Press q to exit")
                self.renderEngine.wait() # consumes for the world to see.
            elif self.verbose:
                print("Press enter to continue. Press ctrl+c to exit.")
                input()
            if self.replaysavepath:
                self.__save_state(game)
            turn += 1
        winner = game.getWinner()
        print('Winner:', winner)
        return game.getWinner()

    def start(self):
        #Retrieves the port and verbose flag from arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('port', type=int, help='The port to listen on')
        parser.add_argument('--replayloadpath', type=str, help='If selected, outputs the game state to a file.', nargs='?', const=None)
        parser.add_argument('--replaysavepath', type=str, help='The port to listen on', nargs='?', const=None)
        parser.add_argument('--verbose', help='Should display the game turn by turn', action='store_true')
        parser.add_argument('--render', help='Should render the game\'s images turn by turn', action='store_true')

        args = parser.parse_args()
        self.verbose = args.verbose
        self.does_render = args.render
        self.load_path = args.replayloadpath
        self.replaysavepath = args.replaysavepath
        
        if not self.load_path:
            self.__loadMap()
            if self.replaysavepath:
                self.log = {"map": self.map.serialize(), "state":[]}
            self.__launchServer(args.port)
            for player in range(self.NUMPLAYERS):
                self.__connectNextPlayer()
            if self.does_render:
                self.renderEngine = RenderingEngine(self.renderFactory, self.map)
            self.map.print_map()
            self.__runGameLoop(self.map)
            if self.replaysavepath:
                with open(self.replaysavepath, 'w') as outfile:
                    json.dump(self.log, outfile)
                print("Saved to file:", self.replaysavepath)
            for connection in self.connections:
                connection.sock.close()
            self.sock.close()
        else:
            # Load file as json
            # render
            loaded = None
            with open(self.load_path) as f:
                loaded = json.load(f)
            states = loaded['state'] # get states
            maptext = loaded['map']
            self.__loadMap(maptext)
            top = self.map.map
            bottom = deepcopy(top[:-1])
            bottom.reverse()
            gamemap = top+bottom
            self.renderEngine = RenderingEngine(self.renderFactory, self.map)
            for state in states:
                current_units = state["all_units"]
                for key, unit in current_units.items():
                    current_units[key] = self.gameFactory.deserialize_unit(unit)
                current_misc = state["misc"]
                self.renderEngine.update(gamemap, current_units, current_misc)
                self.renderEngine.draw()
                print("Press enter on the rendered window to continue. Press q to exit")
                self.renderEngine.wait() # consumes for the world to see.