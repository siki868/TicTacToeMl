import numpy as np
import random as rnd
import time
from tensorflow import keras


def print_table(table):
    """
    Just prints it out
    """
    print()
    t = list(table)
    for i in range(len(t)):
        for c in t[i]:
            if c:
                print(' ' + c.upper(), end=' |')
            else:
                print('  ', end=' |')
        print()
        if i < 2:
            print('-' * 12)
    print()

def move(ch, x, y, table):
    """
    x - row
    y - column
    """
    if table[x, y]:
        pass
    else:
        table[x, y] = ch

def bot_random_move(ch, table):
    """
    ch - x ili o
    Plays random moves that are available
    original_move - (i, j) where i and j are row and column
    move - original_move converted to 1D
    """
    # [['x', 'None', 'o', 'x', 'x', 'o', 'None', 'None', 'x']] npr
    t = table.reshape(1, 9)[0]
    # (0, 1), (2, 0), (2, 1)
    indexes_original = [i for i in range(len(t)) if not t[i]]
    indexes = [np.unravel_index(i, (3, 3)) for i in indexes_original]
    i = rnd.randint(0, len(indexes_original)-1)
    move = indexes[i]
    original_move = indexes_original[i]
    table[move[0], move[1]] = ch
    return move, original_move


def generate_table():
    """
    Makes empty table
    """
    return np.full((3,3 ), None)

def check_table(table):
    """
    table - 3x3 table
    Returns x if X wins, o if its O, None if table has open spots to paly, and Draw if its a draw
    """
    for i in range(3):
        row = table[i]
        col = table[:,i]

        if all([a=='x' for a in row]):
            return 'x'
        elif all([a=='o' for a in row]):
            return 'o'
        
        if all([a=='x' for a in col]):
            return 'x'
        elif all([a=='o' for a in col]):
            return 'o'

    main_diag = np.diag(table)
    second_diag = np.diag(np.fliplr(table))

    if all([a=='x' for a in main_diag]):
            return 'x'
    elif all([a=='o' for a in main_diag]):
        return 'o'

    if all([a=='x' for a in second_diag]):
            return 'x'
    elif all([a=='o' for a in second_diag]):
        return 'o'


    if all([a for a in table.reshape(9, 1)]):
        return 'Draw'
    else:
        return None

def get_table_state(table, ch):
    """
    Return state in an arry: None - 0, my - {}, oponent - {}
    """
    t = table.reshape((1, 9))[0]
    ret = []
    for el in t:
        if el:
            if el == ch:
                ret.append(2)
            else:
                ret.append(1)
        else:
            ret.append(0)

    return np.array(ret).reshape(t.shape)

def get_table_state_default(table):
    """
    Vraca state u obliku niza: None - 0, X - 1, O - 2
    """
    t = table.reshape((1, 9))[0]
    ret = []
    for el in t:
        if el == 'x':
            ret.append(1)
        elif el == 'o':
            ret.append(2)
        else:
            ret.append(0)
        
    return np.array(ret).reshape(t.shape)

def get_random_data_default():
    """
    ** WITH DEFAULT STATE
    Gets data out of random plays by the bot_random_move fun and saves it in files
    valid_moves -  by x and o where they have won the game 
    valid_states - tables states of all the moves he has won (same length as valid_moves)
    """
    # states = []
    valid_x_moves = []
    valid_o_moves = []
    valid_x_states = []
    valid_o_states = []
    for j in range(20000):
        table = generate_table()
        i = 0
        state = None
        x_moves = []
        o_moves = []
        x_states = []
        o_states = []
        state = check_table(table)
        while not state:
            table_state = get_table_state_default(table)
            if i%2 == 0:
                x_moves.append(bot_random_move('x', table)[1])
                x_states.append(table_state)
            else:
                o_moves.append(bot_random_move('o', table)[1])
                o_states.append(table_state)

            state = check_table(table)
            # x, y = [int(i) for i in input().split()]
            # move('x', x, y, table)
            # print_table(table)
            # time.sleep(1)
            i += 1
        if state == 'x':
            valid_x_moves.extend(x_moves)
            valid_x_states.extend(x_states)
        elif state == 'o':
            valid_o_moves.extend(o_moves)
            valid_o_states.extend(o_states)
        print(j)
        # states.append(state)

    print(len(valid_x_moves), len(valid_x_states), len(valid_o_moves), len(valid_o_states))
    valid_x_moves = np.array(valid_x_moves)
    valid_x_states = np.array(valid_x_states)
    valid_o_moves = np.array(valid_o_moves)
    valid_o_states = np.array(valid_o_states)
    np.save('x_moves.npy', valid_x_moves)
    np.save('x_states.npy', valid_x_states)
    np.save('o_moves.npy', valid_o_moves)
    np.save('o_states.npy', valid_o_states)

def get_random_data():
    """
    Gets data out of random plays by the bot_random_move fun and saves it in files
    valid_moves -  by x and o where they have won the game 
    valid_states - tables states of all the moves he has won (same length as valid_moves)
    """
    # states = []
    valid_x_moves = []
    valid_o_moves = []
    valid_x_states = []
    valid_o_states = []
    for j in range(20000):
        table = generate_table()
        i = 0
        state = None
        x_moves = []
        o_moves = []
        x_states = []
        o_states = []
        state = check_table(table)
        while not state:
            if i%2 == 0:
                table_state = get_table_state(table, 'x')
                x_moves.append(bot_random_move('x', table)[1])
                x_states.append(table_state)
            else:
                table_state = get_table_state(table, 'o')
                o_moves.append(bot_random_move('o', table)[1])
                o_states.append(table_state)

            state = check_table(table)
            # x, y = [int(i) for i in input().split()]
            # move('x', x, y, table)
            # print_table(table)
            # time.sleep(1)
            i += 1
        if state == 'x':
            valid_x_moves.extend(x_moves)
            valid_x_states.extend(x_states)
        elif state == 'o':
            valid_o_moves.extend(o_moves)
            valid_o_states.extend(o_states)
        print(j)
        # states.append(state)

    print(len(valid_x_moves), len(valid_x_states), len(valid_o_moves), len(valid_o_states))
    valid_x_moves = np.array(valid_x_moves)
    valid_x_states = np.array(valid_x_states)
    valid_o_moves = np.array(valid_o_moves)
    valid_o_states = np.array(valid_o_states)
    np.save('x_moves.npy', valid_x_moves)
    np.save('x_states.npy', valid_x_states)
    np.save('o_moves.npy', valid_o_moves)
    np.save('o_states.npy', valid_o_states)


def play():
    """
    Play against AI
    """
    i = 0
    table = generate_table()
    state = check_table(table)
    model = keras.models.load_model('ttt6k.h5')
    while not state:
        table_state = get_table_state_default(table).reshape((1, 9))
        print(table_state)
        if i%2 == 0:
            actions = model.predict(table_state)
            print(actions)
            action = np.argmax(actions)
            print(action)
            x, y = np.unravel_index(action, (3, 3))
            move('x', x, y, table)

            # x, y = [int(i) for i in input().split()]
            # move('x', x, y, table)
        else:
            # actions = model.predict(table_state)
            # print(actions)
            # action = np.argmax(actions)
            # print(action)
            # x, y = np.unravel_index(action, (3, 3))
            # move('o', x, y, table)

            x, y = [int(i) for i in input().split()]
            move('o', x, y, table)
        print_table(table)
        state = check_table(table)
        i += 1

    if state == 'o':
        print('O won the game')
    elif state == 'x':
        print('X won the game')
    else:
        print('Nereseno!')

def bot_play():
    """
    2 bots playing against each other
    """
    i = 0
    table = generate_table()
    state = check_table(table)
    model_a = keras.models.load_model('ttt.h5')
    model_b = keras.models.load_model('ttt1.h5')
    while not state:
        table_state = get_table_state(table).reshape((1, 9))
        if i%2 == 0:
            action = np.argmax(model_b.predict(table_state))
            x, y = np.unravel_index(action, (3, 3))
            move('x', x, y, table)
        else:
            # print(table_state.shape)
            action = np.argmax(model_b.predict(table_state))
            x, y = np.unravel_index(action, (3, 3))
            move('o', x, y, table)
        print_table(table)
        state = check_table(table)
        time.sleep(1.5)
        i += 1

    if state == 'o':
        print('O won the game')
    elif state == 'x':
        print('X won the game')
    else:
        print('Nereseno!')


        
if __name__ == "__main__":
    play()