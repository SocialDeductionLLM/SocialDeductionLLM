import sys
import numpy as np
import random
import torch

from string_env import StringAmogus
from transformers import AutoTokenizer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

pipeline = AutoTokenizer.from_pretrained("RWKV/rwkv-4-world-430m", trust_remote_code=True)

num_players = 5
num_imposters = 1
rewards = np.array([0] * num_players)
num_games = 1 #num_worlds // (num_players - num_imposters)

win_types = [0, 0, 0, 0]

game_lengths = []

total_all_generations = []
game_num = 0
all_lens = 0
for _ in range(1):
    env = StringAmogus(num_players=num_players, num_imposters=num_imposters, num_observers=0, discussion_turns=8)
    # print({str(pipeline.encode(" " + chr(ord('B') + i))): env.all_mc_actions[i] for i in range(len(env.all_mc_actions))})
    # print(env.player_state[0].name, end=" ")
    player_to_view = 0

    def generator():
        while not env.done:
            yield


    partial_generations = []
    all_generations = ["" for _ in range(env.num_players)]
    did_vote=False
    t = 0
    for _ in generator():
        env.step()
        for i in range(env.num_players):
            all_generations[i] += env.observations[i]
            if env.actives[i]:
                if len(env.valid_actions[i]) == 0:
                    env.actions[i] = " 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19\n"
                elif env.ask_probs[i]:
                    env.actions[i] = env.valid_actions[i][random.randrange(0, len(env.valid_actions[i]))]
                    env.action_probs[i] = [0.2] * 7
                    did_vote = True
                else:
                    env.actions[i] = env.valid_actions[i][random.randrange(0, len(env.valid_actions[i]))]
            else:
                env.actions[i] = ""
            all_generations[i] += env.actions[i]
        t += 1
    if did_vote:
        for i in range(env.num_players):# - 1):
            true_answer = env.player_names[0]
            if env.player_state[i].alive:
                partial_generations.append(all_generations[i])
    game_num += 1

    total_all_generations.extend(partial_generations)

    if len(partial_generations) == 0:
        continue
    else:
        print(partial_generations[0])
        generation_lengths = [len(pipeline.encode(x)) for x in all_generations]
        game_lengths.append(max(generation_lengths))
        rewards += np.array(env.rewards)
        win_types[env.win_type] += 1
        print(generation_lengths)
        all_lens += sum(generation_lengths)
        # break

print(all_lens)

# print(generation_lengths)
