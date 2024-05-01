#!/usr/bin/env python3

import numpy as np
import random
import sys
from multiprocessing import Pool
from tqdm import tqdm
import os
from functools import partial
from scipy import stats

def gamblers_markov(X, Y, W, p):
    # define our matrix shape
    shape = X + Y + 1
    markov = np.zeros((shape, shape))
    # fill in the absorbing states
    markov[0, 0] = 1
    markov[-1, -1] = 1
    for i in range(1, shape-1):
        # define win and loss situations
        win = min(i + W, X + Y)
        lose = max(i - W, 0)
        # fill in matrix with probabilities
        markov[i, lose] = 1 - p
        markov[i, win] = p
    return markov

def find_probs(matrix):
    # build a fundamental matrix N, with the given formula N = (I - Q)^-1
    # https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/
    # ignore the absorbing states to make Q
    Q = matrix[1:-1, 1:-1]
    # create the identity matrix
    I = np.eye(Q.shape[0])
    # create the fundamental
    N = np.linalg.inv(I - Q)
    # extract the transients (decomposition)
    R = matrix[1:-1, [0, -1]]
    # dot product the fundamental with the transients
    B = np.dot(N, R)
    # find the probabilities from the matrix
    prob_ruin = np.mean(B[:, 0])
    prob_goal = np.mean(B[:, -1])
    return prob_ruin, prob_goal

def gamblers_ruin(X,Y,W,p):
    # make a play
    play = find_probs(gamblers_markov(X,Y,W,p))
    # return them as title and percentage strings
    string_ruin = f"Ruin: {round(play[0] * 100,4)}%"
    string_goal = f"Goal: {round(play[1] * 100,4)}%"
    return string_ruin, string_goal

def gamblers_chance(_):
    # create random entries
    init_cash = random.randint(1, 1000)
    profit_goal = random.randint(1, 1000)
    # pick a random number between 0 and 0.5 (since we are in a casino), excluding 0 and 0.5; 1e-10 = 0.0000000001
    tiny = 1e-10
    prob_win = random.uniform(tiny, 0.5 - tiny)
    # not using gamblers_ruin() here as we can't take an average of strings
    t_ruin, t_goal = find_probs(gamblers_markov(init_cash, profit_goal, 1, prob_win))
    b_ruin, b_goal = find_probs(gamblers_markov(init_cash, profit_goal, init_cash, prob_win))
    return np.array([[t_ruin, t_goal], [b_ruin, b_goal]])

# bring in arguments
def rand_avg_plays(game, runs, *args):
    if args:
        game = partial(game, *args)
    # create a list with our plays, now with pool
    with Pool(16) as pool:
        # use np.array for more math abilities
        play_list = np.array(list(tqdm(pool.imap_unordered(game, [None]*runs), total=runs)))
    # take the average from our playlist
    average_plays = np.mean(play_list, axis=0)
    # get modes too (keepdims false for compatibility with linux's scipy)
    modes = stats.mode(np.round(play_list, 2), axis=0, keepdims=False)
    # and standard deviations
    std_devs = np.std(play_list, axis=0) * 100
    # break down and loop to build output
    output = f"\nAverages after {runs} runs:\n"
    strategies = ['Timid', 'Bold']
    if isinstance(game, partial) and game.func is gamblers_limits:
      strategies.append('Limit')
    outcomes = ['Ruin', 'Goal']
    for i, strategy in enumerate(strategies):
        output += f"  {strategy}:\n"
        for j, outcome in enumerate(outcomes):
            # calculate the means
            avg_val = round(average_plays[i,j] * 100, 4)
            # calculate the modes
            mode_val, mode_freq = modes[0][i,j], modes[1][i,j]
            # because Ruin + Goal are opposites of each other adding up to 1.0 / 100%, we only need to calculate one
            if outcome == 'Ruin':
                # std dev of averages and modal freq will be the same for both Ruin and Goal
                output += f"    Standard Deviation = {round(std_devs[i, j], 4)}% | Modal Frequency = {int(mode_freq)}/{runs}\n"
            # averages and modes will be separate
            output += f"    {outcome}: Average = {avg_val}% | Approx. Mode = {round(mode_val * 100, 3)}%\n"
    print(output)

# like chance, but more restrictive
def gamblers_limits(init_min_lim, init_max_lim, goal_min_lim, goal_max_lim, wager_min_lim, wager_max_lim, _):
    # create random entries with limits
    init_cash = random.randint(init_min_lim, init_max_lim)
    profit_goal = random.randint(goal_min_lim, goal_max_lim)
    wager = random.randint(wager_min_lim, wager_max_lim)
    tiny = 1e-10
    prob_win = random.uniform(tiny, 0.5 - tiny)
    t_ruin, t_goal = find_probs(gamblers_markov(init_cash, profit_goal, 1, prob_win))
    b_ruin, b_goal = find_probs(gamblers_markov(init_cash, profit_goal, init_cash, prob_win))
    l_ruin, l_goal = find_probs(gamblers_markov(init_cash, profit_goal, wager, prob_win))
    return np.array([[t_ruin, t_goal], [b_ruin, b_goal], [l_ruin, l_goal]])

# input prompt to choose which function to use
def game_prompt(*args):
    # numpy + pool don't play nice together, so we need to set these values to singular
    # https://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # decide whether we are in colab or using this as a local script, and determine what game we are playing
    if not args or 'COLAB_GPU' in os.environ and not args:
        game = str(input("Enter 'ruin', 'chance', or 'limits': "))
    elif 'COLAB_GPU' in os.environ:
        game = args[0]
    elif len(sys.argv) > 1:
        game = sys.argv[1]
    else:
        print(f"Usage: {sys.argv[0]} <game> [*args]")
        sys.exit()

    expected = {
        'ruin': ['Starting balance (X >= 1): ', 'Profit goal (Y >= 1): ', 'Wager amount (W >= 1): ', 'Win probability (0. < p < 1.): '],
        'chance': ['Enter number of plays: '],
        'limits': ['Min starting balance (iMin >= 1): ', 'Max starting balance (iMax >= 1): ', 'Min profit goal (gMin >= 1): ', 'Max profit goal (gMax >= 1): ', 'Min wager amount (wMin >= 1): ', 'Max wager amount (wMax >= 1): ', 'Enter number of plays: ']
    }

    if game not in expected:
        print("Ruin: 100%")
        sys.exit()

    game_args = list(args[1:]) if len(args) > 1 else []
    for prompt in expected[game][len(game_args):]:
        game_args.append(input(prompt))

    # input for ruin
    if game == "ruin":
        # perform a check
        if not 0. < float(game_args[3]) < 1.:
            print("p value must be between 0 and 1")
            sys.exit()
        # run a game
        probs_ruin = gamblers_ruin(int(game_args[0]), int(game_args[1]), int(game_args[2]), float(game_args[3]))
        # return the final results
        print(f"""
        Starting Cash: ${game_args[0]} | Profit Goal: +${game_args[1]} | Wager: ${game_args[2]} ({(int(game_args[2]) / int(game_args[0])) * 100}% Strat) | Win Chance: {round(float(game_args[3]) * 100, 2)}%
        {probs_ruin[0]}
        {probs_ruin[1]}
        """)
    # input for chance
    elif game == "chance":
        rand_avg_plays(gamblers_chance, int(game_args[0]))
    # input for limits
    elif game == "limits":
        rand_avg_plays(gamblers_limits, int(game_args[6]), int(game_args[0]), int(game_args[1]), int(game_args[2]), int(game_args[3]), int(game_args[4]), int(game_args[5]))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print("""Gamblers Ruin with Markov Matrices
Written by Oren Klopfer (oklopfer)
Link: https://github.com/oklopfer/ruin/gamblers.py
    """)
    game_prompt() if 'COLAB_GPU' in os.environ else game_prompt(*sys.argv[1:])