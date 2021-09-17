#!/usr/bin/env python3

import argparse
import socket
import os
import random

from grid_fighters import GridFighters
from client_connection import ClientConnection
from tiles import WallTile, GroundTile

MAX_TURNS = 200

#Retrieves the port and verbose flag from arguments
parser = argparse.ArgumentParser()
parser.add_argument('port', type=int, help='The port to listen on')
parser.add_argument('--verbose', help='Should display the game turn by turn', action='store_true')
args = parser.parse_args()

#Creates and connects the socket connection to the clients
sock = socket.socket()
host = socket.gethostname()
sock.bind((host, args.port))

sock.settimeout(7)
sock.listen(2)

print('Waiting for client 1...')
conn1, addr1 = sock.accept()
if not args.verbose:
    conn1.settimeout(3)
p1 = ClientConnection(conn1, 'p1', args.verbose)

print('Waiting for client 2...')
conn2, addr2 = sock.accept()
if not args.verbose:
    conn2.settimeout(3)
p2 = ClientConnection(conn2, 'p2', args.verbose)

#Picks a random map from the maps folder and creates the game
# TODO: CHANGE MAP LOADING SYSTEM
file_name = 'maps/{}'.format(random.choice(os.listdir('maps')))

game = GridFighters(p1, p2, open(file_name, 'r'))
# TODO: Change end game logic
#Ticks the game unit there is a winner or the max_turns is reached
turn = 0
winner = None
print('Game starting...')
while turn < MAX_TURNS and winner == None:
    winner = game.tick(MAX_TURNS - turn)
    turn += 1


print('Winner:', winner)

conn1.close()
conn2.close()
sock.close()
