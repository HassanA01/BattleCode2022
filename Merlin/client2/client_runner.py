from Engine.client.client_launcher import ClientLauncher

from .grid_player import GridPlayer

from .factories import MerlinDecodeDataFactory, MerlinEncodeDataFactory
class MerlinClientLauncher(ClientLauncher):
    '''
        A launch system for the Merlin game Client
    '''
    def getDecodeDataFactory(self):
        return MerlinDecodeDataFactory()

    def getEncodeDataFactory(self):
        return MerlinEncodeDataFactory()

    def getPlayer(self):
        return GridPlayer()
