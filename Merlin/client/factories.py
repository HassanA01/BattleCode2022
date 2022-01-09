from Engine.client.client_launcher import DecodeDataFactory, EncodeDataFactory

from .helper_classes import Map, Units
class MerlinDecodeDataFactory(DecodeDataFactory):
  def decode(self, js):
    return ({
      'game_map': Map(js['map']),
      'your_units': Units(js['your_units']),
      'enemy_units': Units(js['enemy_units']),
      'resources': js['resources'],
      'turns_left': js['turns_left'],
      'your_flag': js['your_flag'],
      'enemy_flag': js['enemy_flag']
    })

class MerlinEncodeDataFactory(EncodeDataFactory):
  def encode(self,moves):
      return moves
