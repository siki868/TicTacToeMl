import numpy as np
import random as rnd
import time

def print_table(table):
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
    if table[x, y]:
        pass
    else:
        table[x, y] = ch

def bot_random_move(ch, table):
    # np.unrabel_index(1D index, (shape))
    t = table.reshape(1, 9)
    indexes = [np.unravel_index(i, (3, 3)) for i in range(len(t[0])) if not t[0, i]]
    move = rnd.choice(indexes)
    table[move[0], move[1]] = ch



def generate_table():
    return np.full((3,3 ), None)

def check_table(table):

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
    elif all([a=='o' for a in second_diag]):
        return 'o'


    if all([a for a in table.reshape(9, 1)]):
        return 'Draw'
    else:
        return None


if __name__ == "__main__":
    table = generate_table()
    # table = np.array([['x', 'o', None], ['o', 'x', 'o'], [None, 'x', 'x']])

    i = 0
    # ch = 'x'
    while not check_table(table):
        if i%2 == 0:
            bot_random_move('x', table)
        else:
            bot_random_move('o', table)

        # x, y = [int(i) for i in input().split()]
        # move('x', x, y, table)
        print_table(table)
        time.sleep(1)
        i += 1

    state = check_table(table)
    if state == 'x':
        print('X je pobedio')
    elif state == 'o':
        print('O je pobedio')
    else:
        print('Nereseno')
