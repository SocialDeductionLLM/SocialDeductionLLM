from dataclasses import dataclass, field
import random

import os

REPORTING_ALIGNMENT = 10

MAX_NUM_PLAYERS = 7

ALL_PLAYER_NAMES = ["Red", "Green", "Orange", "Purple", "Blue", "Yellow", "Pink"]
# ALL_PLAYER_NAMES = ["0", "1", "2", "3", "4", "5", "6"]

class PlayerType:
    IMPOSTER = 0
    CREWMATE = 1
    OBSERVER = 2


class PhaseType:
    INTRO = 0
    GAMEPLAY_INTRO = 1
    GAMEPLAY = 2
    REPORTED = 3
    DISCUSSION_VOLUNTEER = 4
    DISCUSSION_CLAIM = 5
    VOTING = 6
    VOTING_RESULT = 7
    DISCUSSION_SURVEY=8

class WinType:
    MORE_IMPOSTERS = 0
    NO_IMPOSTERS = 1
    NO_TASKS = 2
    TIME_OUT = 3


@dataclass
class TaskState:
    x_loc: int = 0
    y_loc: int = 0
    task_time: int = 0
    done: bool = False


@dataclass
class PlayerState:
    name: int = 0
    ptype: int = PlayerType.IMPOSTER
    alive: bool = True
    killed_recently: bool = False
    x_loc: int = 0
    y_loc: int = 0
    old_x_loc: int = 0
    old_y_loc: int = 0
    timer: int = 0
    kill_timer: int = 0
    subphase: int = 0
    task_list: list[TaskState] = field(default_factory=lambda: [])
    old_msg: str = ""
    task_workon: int = 0
    body_reported: str = ""


class StringAmogus:
    def __init__(self,
                 num_players: int = 4,
                 num_imposters: int = 1,
                 width: int = 3,
                 height: int = 2,
                 num_tasks: int = 5,
                 task_time: int = 5,
                 cooldown_time: int = 20,
                 discussion_turns: int = 10,
                 num_observers: int = 1):
        self.num_players = num_players
        self.num_imposters = num_imposters
        self.num_observers = num_observers
        self.width = width
        self.height = height
        self.num_tasks = num_tasks
        self.task_time = task_time
        self.cooldown_time = cooldown_time

        self.player_names = [f"Player {i}" for i in ALL_PLAYER_NAMES]# + ["Observer"]
        if num_observers > 0:
            self.player_names.append("Observer")

        self.default_x = (width - 1) // 2
        self.default_y = 0

        self.observations = ["" for _ in range(self.num_players + num_observers)]
        self.actions = ["" for _ in range(self.num_players + num_observers)]
        self.actives = [False for _ in range(self.num_players + num_observers)]
        self.valid_actions = [[] for _ in range(self.num_players + num_observers)]
        self.true_valid_actions = [[] for _ in range(self.num_players + num_observers)]
        self.rewards = [0 for _ in range(self.num_players + num_observers)]
        self.done = False
        self.win_type = 0

        self.world_time = 0

        self.max_discussion_turns = discussion_turns
        self.discussion_turns = 0

        self.volunteering_players = []
        self.votes = {}

        self.action_probs = [None for _ in range(self.num_players + num_observers)]
        self.task_rewards = [0 for _ in range(self.num_players + num_observers)]
        self.discussion_rewards = [0 for _ in range(self.num_players + num_observers)]
        self.held_out_rewards = [0 for _ in range(self.num_players + num_observers)]
        self.prior_discussion_value = None
        self.prior_held_out_value = None
        
        self.ask_probs = [False for _ in range(self.num_players + num_observers)]

        self.supervised_answer = None
        
        self.volunteering_players_order = []

        self.all_mc_actions = [f'report body of Player {x}' for x in ALL_PLAYER_NAMES] + [f'kill Player {x}' for x in ALL_PLAYER_NAMES] + [f'do Task'] + [f'go north', 'go south', 'go west', 'go east'] + ['wait']
        self.all_mc_actions = [f" {x}\n" for x in self.all_mc_actions]
        self.to_die = []

        self.speaker = None
        
        self.reset()

    def world_print_all(self, msg):
        for i in range(self.num_players + self.num_observers):
            if self.player_state[i].alive:
                self.observations[i] += f"[{self.world_time}] World (to all): {msg}\n"

    def world_print(self, player, msg):
        self.observations[player] += f"[{self.world_time}] World (to you): {msg}\n"

    def tell_all(self, player, msg):
        for i in range(self.num_players + self.num_observers):
            if self.player_state[i].alive and i != player:
                self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[player].name]} (to all): \"{msg}"

    def ping_player(self, i, valid_tasks, do_shuffle=True):
        option_indices = [self.all_mc_actions.index(t) for t in valid_tasks]
        clean_options = {' ' + chr(ord('B') + x): t.strip() for x, t in zip(option_indices, valid_tasks)}
        
        clean_options = list(clean_options.items())
        random.shuffle(clean_options)
        clean_options = ','.join([": ".join(x) for x in clean_options])
        # clean_options_str = ','.join([f'{x[:2]}:{x[:2]}' for x in clean_options])
        
        self.world_print(i, f"You can perform any of the following actions:{clean_options}")
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.valid_actions[i] = [" " + chr(ord('B') + x) for x in option_indices]
        self.true_valid_actions[i] = valid_tasks
        self.actions[i] = ""
        self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[i].name]} (you):"
        
    def ping_player_direct(self, i, valid_tasks, do_shuffle=True):
        include_skip = [" skip\n"]
        # self.world_print(i, f"You can perform any of the following actions: {real_valid_tasks}")
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.valid_actions[i] = valid_tasks + [" skip"]
        self.true_valid_actions[i] = [x + "\n" for x in self.valid_actions[i]]
        self.actions[i] = ""
        self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[i].name]} (you): Player"
        self.ask_probs[i] = True

    def ping_player_yn(self, i):
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.valid_actions[i] = [" yes", " no"]
        self.true_valid_actions[i] = [" yes\n", " no\n"]
        self.actions[i] = ""
        self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[i].name]} (you):"

    def ping_player_vote_survey(self, i):
        living_ids = [" " + j for j in ALL_PLAYER_NAMES]
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.valid_actions[i] = living_ids
        self.true_valid_actions[i] = living_ids
        self.actions[i] = ""
        self.observations[i] += f"[{self.world_time}] World (to you): Which Player are you most suspicious of?\n"
        self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[i].name]} (you): Player"
        self.ask_probs[i] = True

    def ping_observer_vote_survey(self, i):
        living_ids = [" " + j for j in ALL_PLAYER_NAMES]
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.valid_actions[i] = living_ids
        self.true_valid_actions[i] = living_ids
        self.actions[i] = ""
        self.ask_probs[i] = True

    def player_generate(self, i):
        self.speaker = i
        self.valid_actions[i].clear()
        self.actives[i] = True
        self.actions[i] = ""
        self.observations[i] += f"[{self.world_time}] World (to you): It is your turn to speak. What do you want to say to help Crewmates determine which Player is the Imposter?\n"
        # if self.player_state[i].ptype == PlayerType.IMPOSTER:
            # self.observations[i] += f"[{self.world_time}] World (to you): It is your turn to speak now.\n"
        # else:
            # self.observations[i] += f"[{self.world_time}] World (to you): It is your turn to speak now.\n"
        self.observations[i] += f"[{self.world_time}] {self.player_names[self.player_state[i].name]} (you) saying: \""

    def name_to_idx(self, name): # TODO: SEEMS WRONG
        name_id = self.player_names.index(name)
        for i in range(self.num_players):
            if self.player_state[i].name == name_id:
                return i

    def clear_old_info(self, player):
        self.observations[player] = ""
        self.ask_probs[player] = False
        self.rewards[player] = 0
        self.task_rewards[player] = 0
        self.discussion_rewards[player] = 0
        self.held_out_rewards[player] = 0

    def notify_death(self, player_killed, killer_idx):
        killed_idx = self.name_to_idx(player_killed)
        self.player_state[killed_idx].alive = False
        self.player_state[killed_idx].killed_recently = True
        death_msg = f"{self.player_names[self.player_state[killer_idx].name]} killed you!"
        self.world_print(killed_idx, death_msg)

        bystander_msg = f"You notice {self.player_names[self.player_state[killer_idx].name]} killing {player_killed}!"

        x = self.player_state[killer_idx].x_loc
        y = self.player_state[killer_idx].y_loc
        self.player_state[killed_idx].x_loc = x
        self.player_state[killed_idx].old_x_loc = x
        self.player_state[killed_idx].y_loc = y
        self.player_state[killed_idx].old_y_loc = y
        for i in range(self.num_players):
            # if i != killer_idx and self.player_state[i].x_loc == x and self.player_state[i].y_loc == y and self.player_state[i].old_x_loc == x and self.player_state[i].old_y_loc == y and self.player_state[i].subphase in [0, "wait"] and self.player_state[i].alive:
            if i != killer_idx and self.player_state[i].x_loc == x and self.player_state[i].y_loc == y and self.player_state[i].subphase in [0, "wait", "go"] and self.player_state[i].alive:
                self.world_print(i, bystander_msg)

    def process_actions(self, player):
        p_act = self.actions[player]
        if p_act != "" and len(self.true_valid_actions[player]) != 0:
            true_act = self.true_valid_actions[player][self.valid_actions[player].index(p_act)]
            self.observations[player] += ("\n" if p_act.strip() == true_act.strip()[-1] or len(p_act.strip()) != 1 else ":" + true_act)
            p_act = true_act
        if self.state == PhaseType.GAMEPLAY:
            if p_act == "":
                return

            if self.player_state[player].subphase == 0:
                if p_act.startswith(" do"):
                    self.player_state[player].subphase = "do"
                    for t in range(len(self.player_state[player].task_list)):
                        temp_task = self.player_state[player].task_list[t]
                        if not temp_task.done and temp_task.x_loc == self.player_state[player].x_loc and temp_task.y_loc == self.player_state[player].y_loc:
                            break
                    self.player_state[player].task_workon = t#int(p_act[len(" do Task "):].strip())
                    self.world_print(player, f"You are working on Task {self.player_state[player].task_workon}")
                    self.player_state[player].timer = self.player_state[player].task_list[self.player_state[player].task_workon].task_time
                elif p_act.startswith(" go"):
                    self.player_state[player].subphase = "go"
                    self.player_state[player].old_x_loc = self.player_state[player].x_loc
                    self.player_state[player].old_y_loc = self.player_state[player].y_loc
                    self.player_state[player].x_loc, self.player_state[player].y_loc = self.get_relative_room(self.player_state[player], (p_act[len(" go "):].strip()))
                    self.player_state[player].timer = self.task_time
                elif p_act.startswith(" report"):
                    self.player_state[player].subphase = "report"
                    self.player_state[player].body_reported = p_act[len(" report body of "):].strip()
                elif p_act.startswith(" wait"):
                    self.player_state[player].subphase = "wait"
                    self.player_state[player].timer = 1 #self.task_time * 2
                elif p_act.startswith(" kill"):
                    self.player_state[player].kill_timer = self.cooldown_time
                    player_killed = p_act[len(" kill "):].strip()
                    self.world_print(player, f"You killed {player_killed}")
                    # self.notify_death(player_killed, player)
                    self.to_die.append((player_killed, player))
                    self.task_rewards[player] += 1
        if self.state == PhaseType.DISCUSSION_VOLUNTEER:
            print("INVALID")
            if p_act.startswith(" yes"):
                self.volunteering_players.append(player)
                self.task_rewards[player] += 1
            elif p_act.startswith(" no"):
                self.volunteering_players.append(player)
                self.task_rewards[player] -= 1
        if self.state == PhaseType.DISCUSSION_CLAIM:
            if self.actives[player]:
                self.tell_all(player, p_act)
        if self.state == PhaseType.VOTING_RESULT:
            if self.actives[player]:
                if p_act.startswith(" skip"):
                    self.votes[player] = -1
                else:
                    self.votes[player] = self.name_to_idx("Player " + p_act.strip())
        self.actives[player] = False
        self.valid_actions[player].clear()
        self.true_valid_actions[player].clear()
        self.actions[player] = ""

    def get_room_name(self, x, y):
        return f"({x},{y})"#str(x + y * self.width)

    def get_relative_room(self, pstate, direction):
        x = pstate.x_loc
        y = pstate.y_loc

        if direction == 'north':
            y -= 1
        elif direction == 'west':
            x -= 1
        elif direction == 'east':
            x += 1
        elif direction == 'south':
            y += 1
        else:
            assert False
        return x, y
        
    
    def get_room_loc(self, name):
        return name % self.width, name // self.width

    def process_intro(self):
        living_mask = self.get_living_players_mask()
        self.world_print_all(f"You are playing the game of Among Us on a {self.width}x{self.height} grid. There are {self.num_players} players in total. The current set of players are: {', '.join([self.player_names[i] for i in range(len(ALL_PLAYER_NAMES)) if living_mask[i]])}. Of these players, {self.num_imposters} of them are secretly chosen randomly to be Imposters while the rest are Crewmates.")
        self.world_print_all(f"The Crewmates can win the game one of two ways: either by completing all assigned tasks or by ejecting all Impostors. Each crewmate has to complete {self.num_tasks} tasks. Impostors can likewise win in two ways: either by killing or ejecting all Crewmates.")

        for i in range(self.num_players):
            self.world_print(i, f"You are {self.player_names[self.player_state[i].name]}.")
            if self.player_state[i].ptype == PlayerType.IMPOSTER:
                self.world_print(i, f"You are an Imposter. Try to kill or eject all Crewmates. The current set of imposters are: {', '.join([self.player_names[self.player_state[v].name] for v in range(self.num_players) if self.player_state[v].alive and self.player_state[v].ptype == PlayerType.IMPOSTER])}. Your valid actions outside of a discussion time are of the following forms: 'go x' where x is direction to a neighboring room, 'kill x' where x is a crewmate in the current room, 'report the body of x' if player x is a dead body in the room, or 'wait' which allows you to just wait until you see something new.")
            else:
                self.world_print(i, f"You are a Crewmate. Try to complete all your assigned tasks and eject Imposters. Your valid actions outside of a discussion time are of the following forms: 'go x' where x is direction to a neighboring room, 'do Task' when you can complete a task in the current room, 'report the body of x' if player x is a dead body in the room, or 'wait' which allows you to just wait until you see something new.")
        self.state = PhaseType.GAMEPLAY_INTRO

    def get_living_players_mask(self):
        name_mask = [False for _ in range(len(ALL_PLAYER_NAMES))]
        for i in range(self.num_players):
            if self.player_state[i].alive:
                name_mask[self.player_state[i].name] = True
        return name_mask

    def get_room_player_description(self, player, return_stationary=False):
        stationary_players = []
        entering_players = []
        leaving_players = []
        dead_players = []
        x = self.player_state[player].x_loc
        y = self.player_state[player].y_loc
        for i in range(self.num_players):
            if i != player:
                if self.player_state[i].x_loc == x and self.player_state[i].y_loc == y and self.player_state[i].old_x_loc == x and self.player_state[i].old_y_loc == y:
                    if not self.player_state[i].alive:
                        if self.player_state[i].killed_recently:
                            dead_players.append(i)
                    else:
                        stationary_players.append(i)
                elif self.player_state[i].x_loc == x and self.player_state[i].y_loc == y:
                    entering_players.append(i)
                elif self.player_state[i].old_x_loc == x and self.player_state[i].old_y_loc == y:
                    leaving_players.append(i)

        stationary_players.sort(key=lambda v: self.player_state[v].name)
        entering_players.sort(key=lambda v: self.player_state[v].name)
        leaving_players.sort(key=lambda v: self.player_state[v].name)

        msg = ""
        if len(stationary_players) != 0:
            msg += f" You see {', '.join([self.player_names[self.player_state[i].name] for i in stationary_players])} in the room."

        for e in entering_players:
            msg += f" You see {self.player_names[self.player_state[e].name]} entering from room {self.get_room_name(self.player_state[e].old_x_loc, self.player_state[e].old_y_loc)}."
        for e in leaving_players:
            msg += f" You see {self.player_names[self.player_state[e].name]} leaving to room {self.get_room_name(self.player_state[e].x_loc, self.player_state[e].y_loc)}."

        dead_player_names = [self.player_names[self.player_state[i].name] for i in dead_players]
        if len(dead_players) != 0:
            msg += f" You see the dead body of {', '.join(dead_player_names)} in the room!"

        if msg == "":
            msg = " You do not see any other players in the room."

        if return_stationary:
            return msg, dead_player_names, [self.player_names[self.player_state[i].name] for i in stationary_players + entering_players + leaving_players]
        return msg, dead_player_names

    def get_room_task_list(self, player):
        uncompleted_tasks = []
        for t, task in enumerate(self.player_state[player].task_list):
            if not task.done and task.x_loc == self.player_state[player].x_loc and task.y_loc == self.player_state[player].y_loc:
                uncompleted_tasks.append(f"Task {t}")
        return uncompleted_tasks

    def get_innocent_players_mask(self):
        name_mask = [False for _ in range(len(ALL_PLAYER_NAMES))]
        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                name_mask[self.player_state[i].name] = True
        return name_mask

    def process_gameplay_intro(self):
        living_mask = self.get_living_players_mask()
        self.world_print_all(f"The currently alive players are: {', '.join([self.player_names[i] for i in range(len(ALL_PLAYER_NAMES)) if living_mask[i]])}.")
        self.state = PhaseType.GAMEPLAY
        for i in range(self.num_players):
            if self.player_state[i].alive:
                self.player_state[i].subphase = 0
                self.player_state[i].timer = 0
                self.player_state[i].x_loc = self.default_x
                self.player_state[i].y_loc = self.default_y
                self.player_state[i].old_x_loc = self.default_x
                self.player_state[i].old_y_loc = self.default_y
                if self.player_state[i].ptype == PlayerType.CREWMATE:
                    uncompleted_tasks = []
                    for t, task in enumerate(self.player_state[i].task_list):
                        if not task.done:
                            uncompleted_tasks.append(f"Task {t} in room {self.get_room_name(task.x_loc, task.y_loc)}")
                    if len(uncompleted_tasks) > 0:
                        self.world_print(i, f"Your list of uncompleted tasks are: {', '.join(uncompleted_tasks)}. Recall that you can do tasks in any order.")
                    else:
                        self.world_print(i, f"All your tasks are done. Try to determine who the imposters are now!")
                else:
                    self.player_state[i].kill_timer = self.cooldown_time
                    self.world_print(i, f"You are back to gameplay. You have {self.player_state[i].kill_timer} seconds before you can kill again.")

    def get_neighboring_rooms(self, i):
        x = self.player_state[i].x_loc
        y = self.player_state[i].y_loc

        lst = []
        if y > 0:
            lst.append("north")
        if x > 0:
            lst.append("west")
        if x < self.width - 1:
            lst.append("east")
        if y < self.height - 1:
            lst.append("south")
        return lst

    def process_crewmate_game(self, i):
        if self.player_state[i].subphase in [0, 'wait']:
            room_mask, dead_player_names = self.get_room_player_description(i)
            msg = f"You are in room {self.get_room_name(self.player_state[i].x_loc, self.player_state[i].y_loc)}.{room_mask}" 
            room_task_list = self.get_room_task_list(i)
            if len(room_task_list) > 0:
                msg += f" You have the following tasks in this room: {', '.join(room_task_list)}."
            else:
                msg += " You have no tasks in this room."
            if self.player_state[i].old_msg != msg:
                self.player_state[i].subphase = 0
            if self.player_state[i].subphase == 0:
                self.player_state[i].old_msg = msg
                self.world_print(i, msg)
                self.ping_player(i, ([" do Task\n"] if len(room_task_list) > 0 else []) + [f" go {x}\n" for x in self.get_neighboring_rooms(i)] + [f" report body of {x}\n" for x in dead_player_names] + [" wait\n"])

        if self.player_state[i].timer > 0:
            self.player_state[i].timer -= 1
        else:
            if self.player_state[i].subphase == "do":
                self.player_state[i].task_list[self.player_state[i].task_workon].done = True
                self.world_print(i, "You finished the task.")
                self.task_rewards[i] += 1
            elif self.player_state[i].subphase == "go":
                self.player_state[i].old_x_loc = self.player_state[i].x_loc
                self.player_state[i].old_y_loc = self.player_state[i].y_loc
                self.world_print(i, "You finished walking.")
            self.player_state[i].subphase = 0

    def process_imposter_game(self, i):
        if self.player_state[i].subphase in [0, 'wait']:
            room_mask, dead_player_names, stationary_player_names = self.get_room_player_description(i, True)
            msg = f"You are in room {self.get_room_name(self.player_state[i].x_loc, self.player_state[i].y_loc)}.{room_mask}" 
            if self.player_state[i].kill_timer > 0:
                msg += f" You have to wait at least {self.player_state[i].kill_timer} seconds before you can kill again."
            else:
                msg += " You can kill anyone in the room now."
            if self.player_state[i].old_msg != msg:
                self.player_state[i].subphase = 0
            if self.player_state[i].subphase == 0:
                self.player_state[i].old_msg = msg
                self.world_print(i, msg)
                options = [f" go {x}\n" for x in self.get_neighboring_rooms(i)] + [" wait\n"] # + [f" report body of {x}\n" for x in dead_player_names] + [" wait\n"]
                if self.player_state[i].kill_timer == 0:
                    options += [f" kill {x}\n" for x in stationary_player_names]
                self.ping_player(i, options)

        if self.player_state[i].kill_timer > 0:
            self.player_state[i].kill_timer -= 1

        if self.player_state[i].timer > 0:
            self.player_state[i].timer -= 1
        else:
            if self.player_state[i].subphase == "go":
                self.player_state[i].old_x_loc = self.player_state[i].x_loc
                self.player_state[i].old_y_loc = self.player_state[i].y_loc
                self.world_print(i, "You finished walking.")
            self.player_state[i].subphase = 0

    def process_gameplay(self):
        for i in range(self.num_players):
            if self.player_state[i].subphase == "report" and self.player_state[i].alive:
                self.state = PhaseType.REPORTED
                self.world_print_all(f"{self.player_names[self.player_state[i].name]} discovered the dead body of {self.player_state[i].body_reported} in room {self.get_room_name(self.player_state[i].x_loc, self.player_state[i].y_loc)}.")
                self.task_rewards[i] += 1
                return

        for i in range(self.num_players):
            if self.player_state[i].alive:
                if self.player_state[i].ptype == PlayerType.IMPOSTER:
                    self.process_imposter_game(i)
                else:
                    self.process_crewmate_game(i)
        # self.done = True

    def process_reporting(self):
        if self.world_time % REPORTING_ALIGNMENT != 0:
            return
        living_mask = self.get_living_players_mask()
        self.world_print_all(f"The discussion period is beginning. If you have important information, now is the best time to share it. There will be 2 messages sent by each player during this period, followed by a voting period where you can choose a person to eject or withold your vote. As a reminder, the currently alive players are: {', '.join([self.player_names[i] for i in range(len(ALL_PLAYER_NAMES)) if living_mask[i]])}.")
        self.discussion_turns = 0
        self.volunteering_players_order = []
        for i in range(self.num_players):
            if self.player_state[i].killed_recently:
                self.player_state[i].killed_recently = False
            elif self.player_state[i].alive:
                self.volunteering_players_order.append(i)

        random.shuffle(self.volunteering_players_order)

        # self.prior_discussion_value = None
        # self.prior_held_out_value = None
        self.state = PhaseType.DISCUSSION_SURVEY
        self.ping_discussion_survey()

    def ping_discussion_volunteer(self):
        for i in range(self.num_players):
            if self.player_state[i].alive:
                self.world_print(i, "Do you want to speak next? Please respond with yes or no.")
                self.ping_player_yn(i)

    def ping_discussion_survey(self):
        for i in range(self.num_players):
            if self.player_state[i].alive:
                self.ping_player_vote_survey(i)

        for i in range(self.num_observers):
            self.ping_observer_vote_survey(i + self.num_players)

    def process_discussion_volunteer(self):
        self.volunteering_players = [i for i in range(self.num_players) if self.player_state[i].alive]
        if len(self.volunteering_players) == 0:
            self.state = PhaseType.VOTING
        else:
            to_speak = self.volunteering_players_order[self.discussion_turns % len(self.volunteering_players_order)] #random.choice(self.volunteering_players)
            self.player_generate(to_speak)
            self.state = PhaseType.DISCUSSION_CLAIM
            self.volunteering_players = []

    def process_discussion_survey(self):
        # allocate discussion rewards
        true_answer = self.player_state[0].name
        pooled_rewards = 0
        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                pooled_rewards += self.action_probs[i][true_answer]

        if self.prior_discussion_value is None:
            self.prior_discussion_value = pooled_rewards
            for i in range(self.num_players):
                if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                    self.task_rewards[i] = pooled_rewards-1
                elif self.player_state[i].alive and self.player_state[i].ptype == PlayerType.IMPOSTER:
                    self.task_rewards[i] = -(pooled_rewards-1)

        for i in range(self.num_players):
            if i != self.speaker:
                continue
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                self.discussion_rewards[i] = pooled_rewards - self.prior_discussion_value
            elif self.player_state[i].alive and self.player_state[i].ptype == PlayerType.IMPOSTER:
                self.discussion_rewards[i] = -(pooled_rewards - self.prior_discussion_value)

        self.prior_discussion_value = pooled_rewards

        held_pooled_rewards = 0
        for i in range(self.num_observers):
            # print(true_answer, self.action_probs[i+self.num_players])
            if len(self.action_probs[i + self.num_players]) != 0:
                held_pooled_rewards += self.action_probs[i + self.num_players][true_answer]

        if self.prior_held_out_value is None:
            self.prior_held_out_value = held_pooled_rewards

        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                self.held_out_rewards[i] += (held_pooled_rewards - self.prior_held_out_value) #/self.num_players
            elif self.player_state[i].alive and self.player_state[i].ptype == PlayerType.IMPOSTER:
                self.held_out_rewards[i] += -(held_pooled_rewards - self.prior_held_out_value) #/self.num_players

        # print("PRIOR HELD OUT", self.prior_held_out_value, held_pooled_rewards)

        self.prior_held_out_value = held_pooled_rewards        
        
        self.discussion_turns += 1

        if self.discussion_turns > len(self.volunteering_players_order) * 2 or self.max_discussion_turns == 0:#self.max_discussion_turns:
            self.state = PhaseType.VOTING
        else:
            # self.state = PhaseType.DISCUSSION_VOLUNTEER
            # self.ping_discussion_volunteer()
            self.process_discussion_volunteer()

    def process_discussion_claim(self):
        self.state = PhaseType.DISCUSSION_SURVEY
        self.ping_discussion_survey()

    def process_voting_intro(self):
        living_mask = self.get_living_players_mask()
        self.world_print_all(f"The voting period is beginning. You can vote 'skip' or eject one of the currently alive players. What will you vote?") # : {', '.join([self.player_names[i] for i in range(len(ALL_PLAYER_NAMES)) if living_mask[i]])}
        for i in range(self.num_players):
            if self.player_state[i].alive:
                self.ping_player_direct(i, [f" {ALL_PLAYER_NAMES[i]}" for i in range(len(ALL_PLAYER_NAMES)) if living_mask[i]])
        self.state = PhaseType.VOTING_RESULT
        self.votes = {}

    def check_win_conditions(self):
        if self.check_if_crewmate_win():
            self.world_print_all("All imposters have been ejected! Crewmates win.")
            self.done = True
            for i in range(self.num_players):
                if self.player_state[i].ptype == PlayerType.CREWMATE:
                    self.rewards[i] = 1
                    self.task_rewards[i] = sum([1 for t in self.player_state[i].task_list if not t.done])
            self.win_type = WinType.NO_IMPOSTERS
            return
        elif self.check_if_more_imposters():
            self.world_print_all("There are currently more imposters than crewmates. Imposters win!")
            self.done = True
            for i in range(self.num_players):
                if self.player_state[i].ptype == PlayerType.IMPOSTER:
                    self.rewards[i] = 1
            self.win_type = WinType.MORE_IMPOSTERS
            return
        elif self.check_no_remaining_tasks():
            self.world_print_all("All tasks are done! Crewmates win.")
            self.done = True
            for i in range(self.num_players):
                if self.player_state[i].ptype == PlayerType.CREWMATE:
                    self.rewards[i] = 1
            self.win_type = WinType.NO_TASKS
            return

    def check_if_crewmate_win(self):
        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.IMPOSTER:
                return False
        return True

    def check_if_more_imposters(self):
        num_imposters = 0
        num_crewmates = 0
        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.IMPOSTER:
                num_imposters += 1
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                num_crewmates += 1
        return num_imposters >= num_crewmates

    def check_no_remaining_tasks(self):
        for i in range(self.num_players):
            if self.player_state[i].alive and self.player_state[i].ptype == PlayerType.CREWMATE:
                for t in self.player_state[i].task_list:
                    if not t.done:
                        return False
        return True

    def process_votes(self):
        inverted_dictionary = {}
        for i, vote in self.votes.items():
            if vote not in inverted_dictionary:
                inverted_dictionary[vote] = [i]
            else:
                inverted_dictionary[vote] += [i]

        best = -1
        best_count = -1
        for i, voters in inverted_dictionary.items():
            voters.sort(key=lambda v: self.player_state[v].name)
            if len(voters) > best_count:
                best = i
                best_count = len(voters)
            elif len(voters) == best_count:
                best = -1
        inverted_list = list(inverted_dictionary.items())
        inverted_list.sort(key=lambda v: -1 if v[0] == -1 else self.player_state[v[0]].name)

        messages = []
        for i, lst in inverted_list:
            if i == -1:
                messages.append(f"skip received {len(lst)} votes")
            else:
                messages.append(f"{self.player_names[self.player_state[i].name]} received {len(lst)} votes")
                if i == 0:
                    for voter in lst:
                        self.task_rewards[voter] += 1

        msg = ", ".join(messages) + "."
        if best == -1:
            msg += " Therefore, nobody is ejected this round."
        else:
            msg += f" Therefore, {self.player_names[self.player_state[best].name]} is ejected this round."
            self.player_state[best].alive = False
        self.world_print_all(msg)
        # self.world_print_all(f"{self.votes}")
        self.check_win_conditions()
        if not self.done:
            # self.win_type = WinType.TIME_OUT
            #  self.done = True
            self.prior_discussion_value = None
            self.state = PhaseType.GAMEPLAY_INTRO

    def process_state(self):
        self.check_win_conditions()
        if self.done:
            return

        if self.state == PhaseType.INTRO:
            self.process_intro()
        elif self.state == PhaseType.GAMEPLAY_INTRO:
            self.process_gameplay_intro()
        elif self.state == PhaseType.GAMEPLAY:
            self.process_gameplay()
        elif self.state == PhaseType.REPORTED:
            self.process_reporting()
        elif self.state == PhaseType.DISCUSSION_SURVEY:
            self.process_discussion_survey()
        elif self.state == PhaseType.DISCUSSION_VOLUNTEER:
            self.process_discussion_volunteer()
        elif self.state == PhaseType.DISCUSSION_CLAIM:
            self.process_discussion_claim()
        elif self.state == PhaseType.VOTING:
            self.process_voting_intro()
        elif self.state == PhaseType.VOTING_RESULT:
            self.process_votes()

    def step(self):
        for i in range(self.num_players + self.num_observers):
            self.clear_old_info(i)
        self.to_die.clear()
        for i in range(self.num_players):
            if self.player_state[i].alive:
                self.process_actions(i)
            else:
                self.actives[i] = False
                self.valid_actions[i].clear()
        for player_killed, killer_idx in self.to_die:
            self.notify_death(player_killed, killer_idx)

        for i in range(self.num_observers):
            self.process_actions(self.num_players + i)
            
        self.process_state()
        self.world_time += 1

    def reset(self):
        self.player_state = [PlayerState() for _ in range(self.num_players)]
        self.state = PhaseType.INTRO

        names = list(range(len(ALL_PLAYER_NAMES)))
        random.shuffle(names)

        for i in range(self.num_players):
            self.player_state[i].name = names[i]
            self.player_state[i].x_loc = self.default_x
            self.player_state[i].y_loc = self.default_y
            if i < self.num_imposters:
                self.player_state[i].ptype = PlayerType.IMPOSTER
                self.supervised_answer = names[i]
            else:
                self.player_state[i].ptype = PlayerType.CREWMATE
                for _ in range(self.num_tasks):
                    new_task = TaskState(
                        x_loc=random.randrange(self.width),
                        y_loc=random.randrange(self.height),
                        task_time=self.task_time
                    )
                    self.player_state[i].task_list.append(new_task)

        for i in range(self.num_observers):
            observer_player = PlayerState()
            observer_player.name = -1
            observer_player.ptype = PlayerType.OBSERVER
            self.player_state.append(observer_player)
        self.world_time = 0
