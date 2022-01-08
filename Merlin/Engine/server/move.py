#### GAME ENGINE CODE
### MOVE
# transform -> (x,y,direction,magnitude), direction can be game specific [abstract]
# repr -> ()
# isValid -> ()

#### GAME SPECIFIC CODE
# A list of valid moves
# Factory for creating moves -> given a moveset, create a move
### GAME-MOVE  
# transform -> import from GAME ENGINE MOVE, transforms based on direction
# repr -> (), return a string version of the move, the subClass should implemenet this
# isValid -> (game state) check to see if this is a valid move, the subClass should implement this

class Move:
    '''The bare basics functionality needed for a move
    '''

    def __init__(self, unit):
        self.unit = unit  # The unit of the one making the move

    def transform(x, y, direction, magnitude):
        ''' 
        Returns coordinates starting at <x,y> in the direciton of  <direction> with magnitude <magnitude>
        The subclass should implement this
        '''
        pass

    def __repr__():
        '''
        Returns the serialized form of this move
        The subclass should implement this
        '''
        pass
