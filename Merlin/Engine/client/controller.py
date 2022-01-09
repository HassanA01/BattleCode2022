import json
from socket import socket


class SocketClosed(Exception):
    pass


class Controller:
    def tick(self, connection: socket, player):
        raise NotImplementedError('Should have implemented this')


class NetworkedController(Controller):
    def __init__(self, connection: socket, player, encodeDataFactory, decodeDataFactory):
        self.conn = connection
        self.player = player
        self.encodeDataFactory = encodeDataFactory
        self.decodeDataFactory = decodeDataFactory

    def safe_recv(self, size: int) -> bytes:
        tmp = self.conn.recv(size)
        if tmp == b'':
            raise SocketClosed()
        return tmp

    def tick(self) -> None:
        try:
            size = int(self.safe_recv(10).decode())
            response = self.safe_recv(size).decode()
            js = json.loads(response)
            parsedData = self.decodeDataFactory.decode(js)
            moves = []
            try:
                moves = self.player.tick(**parsedData)
            except Exception as e:
                
                print("Error in client move generation, sending []")
                print("Error was:", e)
            data = self.encodeDataFactory.encode(moves)
            body = json.dumps(data).encode()
            self.conn.sendall('{:10}'.format(len(body)).encode())
            self.conn.sendall(body)

            return True

        except SocketClosed:
            return False
