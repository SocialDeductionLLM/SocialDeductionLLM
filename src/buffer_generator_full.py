import string_env
from torch.distributions.categorical import Categorical
import random
import torch
import math

from rwkv_fast.modeling_rwkv import RwkvCausalLMOutput

from termcolor import colored

from dataclasses import dataclass

from tqdm import tqdm

import time

from string_env import MAX_NUM_PLAYERS

PAD_FLAG = 0
OBS_FLAG = 1
ACTION_FLAG = 2
SPEAKING_ACTION_FLAG = 3

class DummyConfig:
    def __init__(self):
        self.hidden_size=1
        self.num_hidden_layers=1

class DummyModel:

    def __init__(self, is_imposter=False):
        self.config = DummyConfig()
        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.logits = torch.ones((1, 1, 65536), device='cuda', dtype=self.dtype)
        if is_imposter:
            self.logits[:, :, 301:301+MAX_NUM_PLAYERS] = -1
        else:
            self.logits[:, :, 301:301+MAX_NUM_PLAYERS] = 10000

    def forward(self, input_ids, state=False, use_cache=True, lengths=0, valid_actions=None):
        return RwkvCausalLMOutput(loss=None, logits=self.logits.repeat(input_ids.shape[0], input_ids.shape[1], 1), state=state, hidden_states=None, attentions=None)


class HybridModel:

    def __init__(self, action_model, language_model):
        self.action_model = action_model
        self.language_model = language_model
        self.config = language_model.config
        self.dtype = torch.bfloat16
        self.device = "cuda"

    def forward(self, input_ids, state=False, use_cache=True, lengths=0, valid_actions=None):
        output = self.language_model.forward(input_ids=input_ids, state=state, use_cache=use_cache, lengths=lengths, valid_actions=valid_actions)
        output.logits[:, :, 315:321] = -10000
        mask = [x is not None and len(x) > 0 for x in valid_actions]
        if not any(mask):
            return output
        masked_output = self.action_model.forward(input_ids=input_ids[mask], state=None, use_cache=use_cache, lengths=None)
        output.logits[mask] = masked_output.logits
        return output

class HybridModel2:

    def __init__(self, action_model, language_model):
        self.action_model = action_model
        self.language_model = language_model
        self.config = language_model.config
        self.dtype = torch.bfloat16
        self.device = "cuda"

    def forward(self, input_ids, state=False, use_cache=True, lengths=0, valid_actions=None):
        output = self.language_model.forward(input_ids=input_ids, state=state, use_cache=use_cache, lengths=lengths, valid_actions=valid_actions)
        mask = [x is not None and [320] in x for x in valid_actions]
        # print(valid_actions, mask)
        if not any(mask):
            return output
        masked_output = self.action_model.forward(input_ids=input_ids[mask], state=None, use_cache=use_cache, lengths=None)
        output.logits[mask] = masked_output.logits
        return output


@dataclass
class Buffer:
    all_tokens: list[int]
    token_flags: list[int] 
    logprobs: list[float]
    rewards: list[float]
    separated_rewards: list[list[float]]
    values: list[float]
    length: int
    sparse_reward_weight: int
    task_reward_weight: int
    discussion_reward_weight: int
    held_out_reward_weight: int

    state: list

    def __init__(self, state_construction_fn, value_idx, sparse_reward_weight=1, task_reward_weight=0.2, discussion_reward_weight=0.1, held_out_reward_weight=0.1):
        self.all_tokens = []
        self.token_flags = []
        self.logprobs = []
        self.rewards = []
        self.separated_rewards = []
        self.values = []
        self.length = 0
        self.sparse_reward_weight = sparse_reward_weight
        self.task_reward_weight = task_reward_weight
        self.discussion_reward_weight = discussion_reward_weight
        self.held_out_reward_weight = held_out_reward_weight
        self.value_idx = value_idx

        self.state = state_construction_fn()

    def extend_obs(self, new_tokens, new_separated_rewards):
        if self.length > 0:
            self.separated_rewards[self.length-1][0] += new_separated_rewards[0]
            self.separated_rewards[self.length-1][1] += new_separated_rewards[1]
            self.separated_rewards[self.length-1][2] += new_separated_rewards[2]
            self.separated_rewards[self.length-1][3] += new_separated_rewards[3]
            self.rewards[self.length-1] = self.separated_rewards[self.length-1][0] * self.sparse_reward_weight + self.separated_rewards[self.length-1][1] * self.task_reward_weight + self.separated_rewards[self.length-1][2] * self.discussion_reward_weight + self.separated_rewards[self.length-1][3] * self.held_out_reward_weight

        self.all_tokens.extend(new_tokens)
        self.token_flags.extend([OBS_FLAG] * len(new_tokens))
        self.logprobs.extend([0] * len(new_tokens))
        self.rewards.extend([0] * len(new_tokens))
        self.separated_rewards.extend([[0.0, 0.0, 0.0, 0.0] for _ in range(len(new_tokens))])
        self.values.extend([0] * len(new_tokens))
        self.length += len(new_tokens)

    def add_action(self, action_token, action_flag, logprob, value):
        self.all_tokens.append(action_token)
        self.token_flags.append(action_flag)
        self.logprobs.append(logprob)
        self.rewards.append(0.0)
        self.separated_rewards.append([0.0, 0.0, 0.0, 0.0])
        self.values.append(value)
        self.length += 1

@dataclass
class TorchBuffer:
    all_tokens: torch.tensor
    token_flags: torch.tensor
    logprobs: torch.tensor
    rewards: torch.tensor
    values: torch.tensor
    advantages: torch.tensor
    returns: torch.tensor
    lengths: torch.tensor
    value_idx: torch.tensor

    def __init__(self, all_buffers, device="cuda"):
        self.all_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(b.all_tokens, device=device) for b in all_buffers], batch_first = True)
        self.token_flags = torch.nn.utils.rnn.pad_sequence([torch.tensor(b.token_flags, device=device) for b in all_buffers], batch_first = True)
        self.logprobs = torch.nn.utils.rnn.pad_sequence([torch.tensor(b.logprobs, device=device) for b in all_buffers], batch_first = True)
        self.rewards = torch.nn.utils.rnn.pad_sequence([torch.tensor(b.rewards, dtype=self.logprobs.dtype, device=device) for b in all_buffers], batch_first = True)
        self.values = torch.nn.utils.rnn.pad_sequence([torch.tensor(b.values, device=device) for b in all_buffers], batch_first = True)
        self.advantages = None #torch.nn.utils.rnn.pad_sequence([torch.tensor(b.rewards, device=device) for b in all_buffers], batch_first = True)
        self.returns = None
        self.lengths = torch.tensor([b.length for b in all_buffers], device=device)
        self.value_idx = torch.tensor([b.value_idx for b in all_buffers], device=device)
        #print(self.all_tokens.dtype, self.token_flags.dtype, self.logprobs.dtype, self.rewards.dtype, self.values.dtype)

    def calculate_advantages_and_returns(self, gamma=0.99, gae_lambda=0.95):
        num_steps = self.all_tokens.shape[1]
        self.advantages = torch.zeros_like(self.logprobs)
        self.returns = torch.zeros_like(self.logprobs)
        did_start = torch.zeros_like(self.logprobs[:, 0], dtype=torch.bool)
        delta = torch.zeros_like(self.logprobs[:, 0])
        lastgaelam = torch.zeros_like(self.logprobs[:, 0])
        oldvalues = torch.zeros_like(self.logprobs[:, 0])

        carry_rewards = torch.zeros_like(self.rewards[:, 0])
        for t in reversed(range(num_steps)):
            chosen_mask = (self.token_flags[:, t] > OBS_FLAG)
            discussion_mask = (self.token_flags[:, t] == SPEAKING_ACTION_FLAG)
            
            carry_rewards[chosen_mask] = 0
            carry_rewards += self.rewards[:, t]
            
            # handle not did_start
            delta[chosen_mask & ~did_start] = carry_rewards[chosen_mask & ~did_start] - self.values[chosen_mask & ~did_start, t]
            self.advantages[chosen_mask & ~did_start, t] = lastgaelam[chosen_mask & ~did_start] = delta[chosen_mask & ~did_start]

            # handle did_start
            discount = torch.ones_like(oldvalues)
            discount[discussion_mask] = 0
            delta[chosen_mask & did_start] = carry_rewards[chosen_mask & did_start] - self.values[chosen_mask & did_start, t] + oldvalues[chosen_mask & did_start] * (1 - (1 - gamma) * discount[chosen_mask & did_start])
            self.advantages[chosen_mask & did_start, t] = lastgaelam[chosen_mask & did_start] = delta[chosen_mask & did_start] + lastgaelam[chosen_mask & did_start] * (1 - (1 - gamma * gae_lambda) * discount[chosen_mask & did_start])

            did_start = did_start | chosen_mask
            oldvalues[chosen_mask & did_start] = self.values[chosen_mask & did_start, t]
            #if torch.any(chosen_mask):
            #    print(self.advantages[:, t], self.advantages[:, t] + self.values[:, t])
        self.returns = self.advantages + self.values


def pre_action_real(worlds, start_idx, end_idx, stepped_env, tokenizer, buffers):
    num_imposters = end_idx - start_idx
    i_next_obs = [[[] for _ in range(num_imposters)] for _ in range(len(worlds))]
    i_valid_actions = [[None for _ in range(num_imposters)] for _ in range(len(worlds))]
    i_next_action_type = [[ACTION_FLAG for _ in range(num_imposters)] for _ in range(len(worlds))]
    for env_idx, env in enumerate(worlds):
        if not stepped_env[env_idx]:
            continue
        for i in range(start_idx, end_idx):
            if len(env.observations[i]) > 0:
                next_obs = tokenizer.encode(env.observations[i])
            else:
                next_obs = []
            i_next_obs[env_idx][i - start_idx] = next_obs
            buffers[env_idx][i-start_idx].extend_obs(next_obs, [env.rewards[i], env.task_rewards[i], env.discussion_rewards[i], env.held_out_rewards[i]])
            if env.actives[i]:
                valid_action = env.valid_actions[i]
                parsed_valid_action = None
                if len(valid_action) == 0:
                    parsed_valid_action = []
                else:
                    parsed_valid_action = [tokenizer.encode(x) for x in valid_action]
                #print("VALID ACTIONS FOR", i, "is", valid_action)
                i_valid_actions[env_idx][i - start_idx] = parsed_valid_action
                if env.ask_probs[i]:
                    i_next_action_type[env_idx][i-start_idx] = tokenizer.encode(" " + str(env.supervised_answer))[0]
                if len(parsed_valid_action) == 0:
                    i_next_action_type[env_idx][i-start_idx] = SPEAKING_ACTION_FLAG
                    
    return i_next_obs, i_valid_actions, i_next_action_type

def post_action_real(worlds, start_idx, end_idx, actions, tokenizer, valid_actions, probs, ppl=None):
    for env_idx, env in enumerate(worlds):
        for i in range(start_idx, end_idx):
            env.actions[i] = tokenizer.decode(actions[env_idx][i-start_idx])
            env.action_probs[i] = probs[env_idx][i-start_idx]
            if ppl is not None:
                env.ppl = ppl[env_idx]


def choose_next_token_real(logits, valid_actions, tok_num, tokenizer):
    if len(valid_actions) == 0:
        if tok_num == 0:
            logits[11-MAX_NUM_PLAYERS] = -10000
        probs = Categorical(logits=logits[MAX_NUM_PLAYERS:])
        action = probs.sample().item() + MAX_NUM_PLAYERS
        decoded_action = tokenizer.decode([action])
        if not decoded_action.isprintable() or decoded_action == '' or tok_num == 19:
            return 11, valid_actions, True, None, probs.log_prob(torch.tensor([11-MAX_NUM_PLAYERS], device=logits.device))
        return action, valid_actions, False, None, probs.log_prob(torch.tensor([action-MAX_NUM_PLAYERS], device=logits.device))
    else:
        possible_starts = list(set([v[0] for v in valid_actions]))
        actual_logits = Categorical(logits=logits[possible_starts])
        perceived_logits = Categorical(logits=logits[MAX_NUM_PLAYERS:])
        probs = actual_logits.probs
        #print(probs, logits[possible_starts], possible_starts, "comes from", tokenizer.decode(possible_starts))
        action_idx = actual_logits.sample()
        action = possible_starts[action_idx]
        new_valid_actions = [v[1:] for v in valid_actions if v[0] == action and len(v) > 1]
        #print(valid_actions, "becomes", new_valid_actions, "updated", (len(new_valid_actions) == 0))
        return action, new_valid_actions, (len(new_valid_actions) == 0), probs, perceived_logits.log_prob(torch.tensor([action-MAX_NUM_PLAYERS], device=logits.device))

def choose_action_real(next_obs, valid_actions, buffers, model, tokenizer, next_action_type, split_value_idx):
    actions = [[[] for _ in range(len(valid_actions[env_idx]))] for env_idx in range(len(valid_actions))]
    probs = [[[] for _ in range(len(valid_actions[env_idx]))] for env_idx in range(len(valid_actions))]

    indices = []
    states = []
    observations = []
    lengths = []
    valid_actions_build = []
    value_idx = []
    for env_idx in range(len(valid_actions)):
        for i in range(len(valid_actions[env_idx])):
            if len(next_obs[env_idx][i]) > 0:
                indices.append((env_idx, i))
                states.append(buffers[env_idx][i].state)
                observations.append(torch.tensor(next_obs[env_idx][i]).to(model.device))
                lengths.append(len(observations[-1]))
                valid_actions_build.append(valid_actions[env_idx][i])
                value_idx.append(split_value_idx[env_idx][i])

    if len(indices) == 0:
        return actions, probs, None, None

    final_indices = []
    final_states = []
    final_observations = []
    final_lengths = []
    for tok_num in range(20):
        merged_states = [torch.cat([states[n][i] for n in range(len(states))]) for i in range(5)]
        input_ids = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
        outputs = model.forward(input_ids=input_ids, state=merged_states, use_cache=True, lengths=torch.tensor(lengths).to(model.device), valid_actions=valid_actions_build)
        logits = outputs.logits[range(len(lengths)), [l - 1 for l in lengths]]
        values = logits[range(len(lengths)), value_idx]
        new_merged_states = outputs.state

        new_states = []
        new_observations = []
        new_lengths = []
        new_indices = []
        new_valid_actions_build = []
        new_value_idx = []
        # Need to update states, observations, lengths, indices, valid_actions_build
        for n, (e, i) in enumerate(indices):
            #print("parsing", (e, i))
            buffers[e][i].state = [new_merged_states[s][n:n+1] for s in range(5)]
            if valid_actions[e][i] is not None:
                #print(valid_actions[e][i], "versus", valid_actions_build[n], "at time", tok_num)
                next_token, cur_valid_actions_build, is_done, prob, logprob = choose_next_token_real(logits[n], valid_actions_build[n], tok_num, tokenizer)
                buffers[e][i].add_action(next_token, next_action_type[e][i] if tok_num != 19 else OBS_FLAG, logprob, values[n])
                #print("AFTER", is_done)
                if tok_num == 0 and prob is not None:
                    probs[e][i] = prob.tolist()
                actions[e][i].append(next_token)
                if not is_done:
                    new_indices.append((e, i))
                    new_states.append(buffers[e][i].state)
                    new_observations.append(torch.tensor([next_token]).to(model.device))
                    new_lengths.append(1)
                    new_valid_actions_build.append(cur_valid_actions_build)
                    new_value_idx.append(split_value_idx[e][i])
                else:
                    #print("REMOVING", (e, i))
                    final_indices.append((e, i))
                    final_states.append(buffers[e][i].state)
                    final_observations.append(torch.tensor([next_token]).to(model.device))
                    final_lengths.append(1)
            elif tok_num == 0:
                probs[e][i] = []#logits[n][0]
        observations = new_observations
        indices = new_indices
        states = new_states
        lengths = new_lengths
        valid_actions_build = new_valid_actions_build
        value_idx = new_value_idx
        if len(indices) == 0:
            break
    return actions, probs, final_indices, final_observations

def get_state(model):
    shape = (1, model.config.hidden_size, model.config.num_hidden_layers)
    state = [
        torch.zeros(
            *shape, dtype=model.dtype if i <= 1 else torch.float32, device=model.device
        )
        for i in range(5)
    ]
    state[4] -= 1e30
    return state

def append_buffers(older_buffer, next_items, do_print=False, tokenizer=None):
    already_printed = False
    for e in range(len(older_buffer)):
        for i in range(len(older_buffer[e])):
            if do_print and len(next_items[e][i]) > 0 and e == 0 and i == 0:#not already_printed:
                print(tokenizer.decode(next_items[e][i]), end="", flush=True)
                already_printed = True
            older_buffer[e][i].extend(next_items[e][i])

def append_rewards(rew_buf, next_obs, next_actions, next_rews):
    for e in range(len(rew_buf)):
        for i in range(len(rew_buf[e])):
            #print("YOOP", rew_buf[e][i], next_rews[e][i])
            if len(rew_buf[e][i]) > 0 and len(next_rews[e][i]) > 0:
                rew_buf[e][i][-1][0] += next_rews[e][i][0]
                rew_buf[e][i][-1][1] += next_rews[e][i][1]
                rew_buf[e][i][-1][2] += next_rews[e][i][2]
            rew_buf[e][i].extend([[0, 0, 0] for _ in range(len(next_obs[e][i]))])
            rew_buf[e][i].extend([[0, 0, 0] for _ in range(len(next_actions[e][i]))])

def prepend_obs(i_next_obs, i_final_indices, i_final_observations):
    if i_final_indices is None:
        return
    #print(i_final_indices, i_final_observations)
    for n, (e, i) in enumerate(i_final_indices):
        i_next_obs[e][i] = [i_final_observations[n].item()] + i_next_obs[e][i]

def get_flattened_buffers(i_buffers, do_calc=True):
    i_buf_flattened = []
    for i_buf in i_buffers:
        i_buf_flattened.extend(i_buf)
    toret = TorchBuffer(i_buf_flattened)
    #print("CALCULATING ADVANTAGES")
    if do_calc:
        toret.calculate_advantages_and_returns()
    #print("CALCULATING DONE")
    return toret

def collect_real_buffers(models, splits, tokenizer, num_worlds, num_players=5, num_imposters=1, world_width=3, min_world_width=None, world_height=2, discussion_turns=6, num_observers=0, discussion_reward_weight=0.1, reporting_alignment=10, num_tasks=5, min_num_tasks=None, randomize=False):
    min_world_width = world_width if min_world_width is None else min_world_width
    min_num_tasks = num_tasks if min_num_tasks is None else min_num_tasks
    string_env.REPORTING_ALIGNMENT = reporting_alignment
    calculated_splits = [0]
    for i in range(len(splits)):
        calculated_splits.append(calculated_splits[i] + splits[i])

    if randomize:
        room_types = [random.randrange(3) for _ in range(num_worlds)] # 1x3, 2x2, 2x3
        all_num_tasks = [random.randrange(3, 6) for _ in range(num_worlds)]
        worlds = [string_env.StringAmogus(
            num_players=num_players,
            num_imposters=1,
            width=2 if room_types[i] == 1 else 3,
            height=1 if room_types[i] == 0 else 2,
            num_tasks=all_num_tasks[i],
            task_time=3,
            cooldown_time=6,
            discussion_turns=discussion_turns,
            num_observers=num_observers
        ) for i in range(num_worlds)]
    else:
        worlds = [string_env.StringAmogus(num_players=num_players, num_imposters=1, width=random.randrange(min_world_width, world_width + 1), height=world_height, num_tasks=random.randrange(min_num_tasks, num_tasks + 1), task_time=3, cooldown_time=5, discussion_turns=discussion_turns, num_observers=num_observers) for _ in range(num_worlds)]

    get_states = [lambda m=m: get_state(m) for m in models]

    buffers = [
        [[Buffer(get_states[i], worlds[j].supervised_answer, discussion_reward_weight=discussion_reward_weight, held_out_reward_weight=discussion_reward_weight*4) for _ in range(splits[i])] for j in range(num_worlds)]
        for i in range(len(models))
    ]

    split_value_idx = [
        [[worlds[j].supervised_answer for _ in range(splits[i])] for j in range(num_worlds)]
        for i in range(len(models))
    ]
    
    num_rounds = 0
    final_indices = [None for _ in models]
    final_observations = [None for _ in models]

    #pbar = tqdm(total=300, desc=" inner loop", position=1)
    
    while True:
        num_rounds += 1
        #pbar.update(1)
        if num_rounds > 300:
            break
        #print(num_rounds)
        all_done = all([world.done for world in worlds])
        if all_done:
            break
        stepped_env = [False for _ in range(num_worlds)]
        for env_idx, env in enumerate(worlds):
            if env.done:
                continue
            env.step()
            stepped_env[env_idx] = True

        for i in range(len(splits)):
            next_obs, valid_actions, next_action_type = pre_action_real(worlds, calculated_splits[i], calculated_splits[i+1], stepped_env, tokenizer, buffers[i])
            
            prepend_obs(next_obs, final_indices[i], final_observations[i])

            actions, probs, final_indices[i], final_observations[i] = choose_action_real(next_obs, valid_actions, buffers[i], models[i], tokenizer, next_action_type, split_value_idx[i])
            post_action_real(worlds, calculated_splits[i], calculated_splits[i+1], actions, tokenizer, valid_actions, probs)

    torch_buffers = [get_flattened_buffers(b) for b in buffers]
    # torch_o_buffers = get_flattened_buffers(o_buffers, do_calc=False)
    sparse_i_wins = [sum([r[0] for r in b[0].separated_rewards]) for b in buffers[0]]
    sparse_c_wins = [sum([r[0] for r in b[0].separated_rewards]) for b in buffers[1]]
    discussion_benefits = [[sum([r[2] for r in bi.separated_rewards]) for bi in b] for b in buffers[1]]
    held_out_benefits = [[sum([r[3] for r in bi.separated_rewards]) for bi in b] for b in buffers[1]]

    # print(tokenizer.decode(torch_o_buffers.all_tokens[0]))
    print("NUM STEPS:", num_rounds)
    return torch_buffers, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits
