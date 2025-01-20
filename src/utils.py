from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions.categorical import Categorical

import numpy as np
import random

import copy

from string_env import MAX_NUM_PLAYERS

RWKV_NAMES = ["RWKV/rwkv-4-world-169m", "RWKV/rwkv-4-world-430m", "RWKV/rwkv-4-world-1b5", "RWKV/rwkv-4-world-3b", "RWKV/rwkv-4-world-7b"]

@dataclass
class Args:
    rwkv_id: int = 2
    """RWKV pretrained LM to start from (0 to 4)"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    seed: int = 0
    """seed of the experiment"""
    other_seed: int = 0
    world_width: int = 3
    """width of the world"""
    world_height: int = 2
    discussion_turns: int = 8
    """width of the world"""

    o_checkpoint_name: str = ""
    i_checkpoint_name: str = ""
    c_checkpoint_name: str = ""
    """Name of checkpoint to reload from"""

    num_imposters: int = 1
    num_crewmates: int = 4
    num_observers: int = 0

    name: str = "Standard"
    """Name of run"""

    use_rl: bool = True
    """Use RL Loss"""
    use_sl: bool = True
    """Use SL Loss"""
    use_wm: bool = False
    """ Use WM Loss"""

    use_bc: bool = False

    # Algorithm specific arguments
    num_iterations: int = 151
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_worlds: int = 10
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    sl_coef: float = 1

    sl_lr: float = 0.0

    discussion_reward_weight:float = 2.0

    kl_coef: float = 0.001

    exp_decay: float = 0.9

    fl_gamma: float = 3

    re_iterations: int = 1

    train_crewmates: bool = True
    train_imposters: bool = True

    num_tasks:int = 5

    ema: float = 0.9

    wm_weight: float = 1.0

    randomize: bool = False

    d_coef: float = 0.0
    a_coef: float = 0.0
    use_d: bool = True

    save: bool = True


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_state(model, num_threads):
    shape = (num_threads, model.config.hidden_size, model.config.num_hidden_layers)
    state = torch.stack([
        torch.zeros(
            *shape, dtype=torch.float32, device=model.device
        )
        for i in range(5)
    ], dim=0)
    state[4] -= 1e30
    return state

def validate_load_success(model_string, c_model):
    with torch.no_grad():
        torch.cuda.empty_cache()
        print("VALIDATING USING TENSOR EXAMPLE")
        example_input = torch.tensor([[11]]).cuda()

        #c_model2 = RwkvForCausalLM.from_pretrained(model_string, torch_dtype=torch.bfloat16, device_map='cuda')

        cm_out = c_model(input_ids=example_input).logits
        print("c_model", cm_out)

def get_batch(size, dat, tok):
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(tok.encode(next(dat)['text']), device='cuda')[:1024] for _ in range(size)], batch_first=True)


def do_pile_training(args, model, dataloader, tok, optimizer):
    batch_size = args.num_worlds * args.num_crewmates // args.update_epochs
    tok_batch = get_batch(batch_size, dataloader, tok)

    # torch.cuda.empty_cache()
    outputs = model.forward(input_ids=tok_batch)

    ans_view = outputs.logits[:, :-1, MAX_NUM_PLAYERS:]
    true_answer = tok_batch[:, 1:]
    
    wm_mask = (true_answer != 0)

    ans_view_restricted_wm = ans_view[wm_mask]
    true_wm_answer_restricted = torch.clamp(true_answer[wm_mask] - MAX_NUM_PLAYERS, min=0)

    cross_entropy_wm = F.cross_entropy(ans_view_restricted_wm, true_wm_answer_restricted, reduction='mean')

    loss = cross_entropy_wm

    all_losses = cross_entropy_wm.item()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return all_losses

def reevaluate_logprobs(args, buf, model, epoch, update_epochs):
    with torch.no_grad():
        states = get_state(model, buf.all_tokens.shape[0] // update_epochs)
        last_logits = torch.zeros((buf.logprobs.shape[0]//update_epochs, 1, 65536), device="cuda", dtype=buf.logprobs.dtype)
        full_token_len = buf.all_tokens.shape[1]
        for start_subset in range(0, full_token_len, 1024):
            # outputs = model.forward(input_ids=buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024], state=states, use_cache=True)
            if torch.all(buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024] == 0):
                break

            outputs = model.forward(input_ids=buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024], state=states, use_cache=True)
            states = outputs.state
            ans_view = torch.cat([last_logits[:, :, MAX_NUM_PLAYERS:], outputs.logits[:, :-1, MAX_NUM_PLAYERS:]], dim=1)

            rl_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] > 1) & (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] <= 3)
            b_actions = buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024][rl_mask] - MAX_NUM_PLAYERS
            if len(b_actions) != 0:
                categorical_dist = Categorical(logits = ans_view[rl_mask])
                newlogprob = categorical_dist.log_prob(b_actions)
            
                oldlogprob = buf.logprobs[epoch::update_epochs, start_subset:start_subset+1024]
                oldlogprob[rl_mask] = newlogprob

            if start_subset + 1023 <= full_token_len:
                last_logits = outputs.logits[:, -1:].detach()

def do_train(args, buf, model, advantages_mean, advantages_std, epoch, do_sl, update_epochs, optimizer, ho_model=None):
    if args.ema > 0:
        ema_model = copy.deepcopy(model)
    num_games = buf.all_tokens.shape[0] // update_epochs
        
    full_token_len = buf.all_tokens.shape[1]
    states = get_state(model, buf.all_tokens.shape[0] // update_epochs)
    last_logits = torch.zeros((buf.logprobs.shape[0]//update_epochs, 1, 65536), device="cuda", dtype=buf.logprobs.dtype)
    if ho_model is not None:
        states_ho = get_state(ho_model, buf.all_tokens.shape[0] // update_epochs)
        last_logits_ho = torch.zeros((buf.logprobs.shape[0]//update_epochs, 1, 65536), device="cuda", dtype=buf.logprobs.dtype)
    all_probs = []
    all_losses = {'bc_loss': 0, 'loss_lm': 0, 'pg_loss': 0, 'entropy_loss': 0, 'v_loss': 0, 'kl_logratio': 0, 'sl_kl_logratio': 0, 'all_probs': []}
    if do_sl:
        all_losses["loss_ans"] = 0

    start_epoch = epoch * num_games
    end_epoch = start_epoch + num_games

    for start_subset in range(0, full_token_len, 1024):
        if torch.all(buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024] == 0):
            break

        if ho_model is not None:
            with torch.no_grad():
                outputs_ho = ho_model.forward(input_ids=buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024], state=states_ho, use_cache=True)
                states_ho = outputs_ho.state.detach()
                ans_view_ho = torch.cat([last_logits_ho[:, :, MAX_NUM_PLAYERS:], outputs_ho.logits[:, :-1, MAX_NUM_PLAYERS:]], dim=1)

        outputs = model.forward(input_ids=buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024], state=states, use_cache=True)
        states = outputs.state.detach()
        ans_view = torch.cat([last_logits[:, :, MAX_NUM_PLAYERS:], outputs.logits[:, :-1, MAX_NUM_PLAYERS:]], dim=1)
        
        # Pure world modeling
        wm_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 1)
        true_wm_answer = buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024] - MAX_NUM_PLAYERS

        ans_view_restricted_wm = ans_view[wm_mask]
        true_wm_answer_restricted = true_wm_answer[wm_mask]

        cross_entropy_wm = F.cross_entropy(ans_view_restricted_wm, true_wm_answer_restricted, reduction='mean')

        sl_kl_logratio = 0

        if do_sl:
            # Pure supervised
            supervised_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] > 3)

            true_supervised_answer = buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024]
            ans_view_restricted_supervised = ans_view[supervised_mask]
            true_supervised_answer_restricted = true_supervised_answer[supervised_mask] - MAX_NUM_PLAYERS

            pure_cross_entropy = F.cross_entropy(ans_view_restricted_supervised, true_supervised_answer_restricted, reduction='none')
            extracted_probabilities = (-pure_cross_entropy).exp()
            all_probs.extend(extracted_probabilities.tolist())
            
            cross_entropy_ans = (pure_cross_entropy * ((1-extracted_probabilities) ** args.fl_gamma))

            if len(cross_entropy_ans) > 0:
                cross_entropy_ans = cross_entropy_ans.mean()
            else:
                cross_entropy_ans = 0

        # Pure RL
        rl_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] > 1) #& (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] <= 3)
        discussion_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 3)
        pure_rl_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 2)
        bc_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 2) | (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 3)

        if not args.use_d:
            rl_mask = pure_rl_mask
        
        b_actions = buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024][rl_mask] - MAX_NUM_PLAYERS
        d_actions = buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024][discussion_mask] - MAX_NUM_PLAYERS
        if len(b_actions) == 0:
            continue
        categorical_dist = Categorical(logits = ans_view[rl_mask])
        entropy = Categorical(logits = ans_view[pure_rl_mask]).entropy().nanmean() if pure_rl_mask.sum() > 0 else 0
        newlogprob, newvalue = categorical_dist.log_prob(b_actions), torch.cat([last_logits[range(last_logits.size(0)), :, buf.value_idx[epoch::update_epochs]], outputs.logits[range(last_logits.size(0)), :-1, buf.value_idx[epoch::update_epochs]]], dim=1)[rl_mask]
        oldlogprob = buf.logprobs[epoch::update_epochs, start_subset:start_subset+1024].clone()


        # if len(d_actions) > 0 and ho_model is not None and not args.use_bc:
        #     categorical_dist_ho = Categorical(logits = ans_view_ho[discussion_mask])
        #     logprob_ho = categorical_dist_ho.log_prob(d_actions)
        #     oldlogprob[discussion_mask] = logprob_ho

        oldlogprob = oldlogprob[rl_mask]

        logratio = newlogprob - oldlogprob
        ratio = torch.clamp(logratio.exp(), max=100)

        if len(d_actions) > 0 and ho_model is not None and args.use_d: # and not args.use_bc:
            categorical_dist_ho = Categorical(logits = ans_view_ho[discussion_mask])
            logprob_ho = categorical_dist_ho.log_prob(d_actions)
            kl_logratio = newlogprob[discussion_mask[rl_mask]] - logprob_ho
            # print(kl_logratio)
            kl_logratio = kl_logratio.square().mean()
        else:
            kl_logratio = 0
        
        mb_advantages = buf.advantages[epoch::update_epochs, start_subset:start_subset+1024][rl_mask]
        if args.norm_adv:
            mb_advantages = (mb_advantages - advantages_mean) / (advantages_std + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2)

        if len(d_actions) > 0:
            pg_loss_a = pg_loss[~discussion_mask[rl_mask]].nanmean()
            pg_loss_d = pg_loss[discussion_mask[rl_mask]].nanmean()
            pg_loss = (0 if pg_loss_a.isnan() else pg_loss_a) * args.a_coef + (0 if pg_loss_d.isnan() else pg_loss_d) * args.d_coef
        else:
            pg_loss = pg_loss.nanmean()

        # if pg_loss > 1000:
        #     print(mb_advantages, ratio)
        #     print(mb_advantages.max(), mb_advantages.min(), ratio.max(), ratio.min())

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - buf.returns[epoch::update_epochs, start_subset:start_subset+1024][rl_mask]) ** 2
            v_clipped = buf.values[epoch::update_epochs, start_subset:start_subset+1024][rl_mask] + torch.clamp(
                newvalue - buf.values[epoch::update_epochs, start_subset:start_subset+1024][rl_mask],
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - buf.returns[epoch::update_epochs, start_subset:start_subset+1024][rl_mask]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.nanmean()
        else:
            # v_loss = 0.5 * ((newvalue - buf.returns[epoch::update_epochs, start_subset:start_subset+1024][rl_mask]) ** 2).nanmean()
            v_loss = 0.5 * ((newvalue - buf.returns[epoch::update_epochs, start_subset:start_subset+1024][rl_mask]) ** 2)
            if len(d_actions) > 0:
                v_loss_a = v_loss[~discussion_mask[rl_mask]].nanmean()
                v_loss_d = v_loss[discussion_mask[rl_mask]].nanmean()
                v_loss = (0 if v_loss_a.isnan() else v_loss_a) * args.a_coef + (0 if v_loss_d.isnan() else v_loss_d) * args.d_coef
            else:
                v_loss = v_loss.nanmean()

        entropy_loss = entropy#.nanmean()
        loss = 0
        if args.use_bc:
            bc_loss = F.cross_entropy(ans_view[bc_mask], true_wm_answer[bc_mask], reduction='none')
            bc_loss = (bc_loss * ((1 - (-bc_loss).exp()) ** args.fl_gamma)).nanmean() - entropy_loss * args.ent_coef
            loss = bc_loss + v_loss * args.vf_coef + kl_logratio * args.kl_coef
        if args.use_rl:
            loss += pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + kl_logratio * args.kl_coef
        # else:
            # loss += v_loss * args.vf_coef + kl_logratio * args.kl_coef
        if args.use_sl and do_sl:
            loss += cross_entropy_ans * args.sl_coef
        if args.use_wm:
            loss += cross_entropy_wm * args.wm_weight

        if do_sl:
            all_losses['loss_ans'] += cross_entropy_ans if isinstance(cross_entropy_ans, int) else cross_entropy_ans.item()
        all_losses['loss_lm'] += cross_entropy_wm.item()
        all_losses['pg_loss'] += pg_loss.item()
        all_losses['entropy_loss'] += entropy_loss if isinstance(entropy_loss, int) else entropy_loss.item()
        all_losses['v_loss'] += v_loss.item()
        if args.use_bc:
            all_losses['bc_loss'] += bc_loss.item()
        all_losses['kl_logratio'] += kl_logratio
        all_losses['sl_kl_logratio'] += sl_kl_logratio

        #print("LOSS IS", loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        if start_subset + 1023 <= full_token_len:
            last_logits = outputs.logits[:, -1:].detach()
            if ho_model is not None:
                last_logits_ho = outputs_ho.logits[:, -1:].detach()

    if args.ema > 0:
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                param.data.copy_(ema_param.data * args.ema + (1 - args.ema) * param.data)
    all_losses['all_probs'] = all_probs
    return all_losses
    


def do_eval(args, buf, model, advantages_mean, advantages_std, epoch, do_sl, update_epochs, optimizer, ho_model=None):
    with torch.no_grad():
        if args.ema > 0:
            ema_model = copy.deepcopy(model)
        num_games = buf.all_tokens.shape[0] // update_epochs

        full_token_len = buf.all_tokens.shape[1]
        states = get_state(model, buf.all_tokens.shape[0] // update_epochs)
        last_logits = torch.zeros((buf.logprobs.shape[0]//update_epochs, 1, 65536), device="cuda", dtype=buf.logprobs.dtype)
        all_probs = []

        start_epoch = epoch * num_games
        end_epoch = start_epoch + num_games

        for start_subset in range(0, full_token_len, 1024):
            if torch.all(buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024] == 0):
                break

            outputs = model.forward(input_ids=buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024], state=states, use_cache=True)
            states = outputs.state.detach()
            ans_view = torch.cat([last_logits[:, :, MAX_NUM_PLAYERS:], outputs.logits[:, :-1, MAX_NUM_PLAYERS:]], dim=1)

            # Pure world modeling
            wm_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] == 1)
            true_wm_answer = buf.all_tokens[epoch::update_epochs, start_subset:start_subset+1024] - MAX_NUM_PLAYERS

            ans_view_restricted_wm = ans_view[wm_mask]
            true_wm_answer_restricted = true_wm_answer[wm_mask]

            cross_entropy_wm = F.cross_entropy(ans_view_restricted_wm, true_wm_answer_restricted, reduction='mean')

            sl_kl_logratio = 0

            if do_sl:
                # Pure supervised
                supervised_mask = (buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024] > 3)

                true_supervised_answer = buf.token_flags[epoch::update_epochs, start_subset:start_subset+1024]
                ans_view_restricted_supervised = ans_view[supervised_mask]
                true_supervised_answer_restricted = true_supervised_answer[supervised_mask] - MAX_NUM_PLAYERS

                pure_cross_entropy = F.cross_entropy(ans_view_restricted_supervised, true_supervised_answer_restricted, reduction='none')
                extracted_probabilities = (-pure_cross_entropy).exp()
                all_probs.extend(extracted_probabilities.tolist())

            if start_subset + 1023 <= full_token_len:
                last_logits = outputs.logits[:, -1:].detach()

        return all_probs
