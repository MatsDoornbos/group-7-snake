# Battlesnake MCTS baseline for the MGAIA BattleSnakes assignment.
# Drop this into main.py and tune the constants near the bottom.

import math
import random
import time
import typing
from collections import defaultdict, deque
from dataclasses import dataclass

DIRS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

TIME_BUDGET = 0.2       # stay below the 1000 ms hard limit
ROLLOUT_DEPTH = 5       # deeper is stronger but slower
C_PUCT = 1.6             # exploration constant for tree policy


def add_pos(pos, move_name):
    dx, dy = DIRS[move_name]
    return (pos[0] + dx, pos[1] + dy)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def softmax_dict(scores, temperature=2.0):
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp((v - m) / max(0.001, temperature)) for k, v in scores.items()}
    z = sum(exps.values())
    if z <= 0:
        p = 1.0 / len(scores)
        return {k: p for k in scores}
    return {k: v / z for k, v in exps.items()}


@dataclass
class SnakeState:
    sid: str
    body: list
    health: int
    alive: bool = True

    def clone(self):
        return SnakeState(self.sid, self.body.copy(), self.health, self.alive)

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)


class State:
    def __init__(self, width, height, snakes, food, hazards, you_id, turn, hazard_damage=14):
        self.width = width
        self.height = height
        self.snakes = snakes
        self.food = set(food)
        self.hazards = set(hazards)
        self.you_id = you_id
        self.turn = turn
        self.hazard_damage = hazard_damage

    @staticmethod
    def from_game_state(game_state):
        width = game_state["board"]["width"]
        height = game_state["board"]["height"]
        you_id = game_state["you"]["id"]
        turn = game_state["turn"]

        settings = game_state.get("game", {}).get("ruleset", {}).get("settings", {})
        hazard_damage = settings.get("hazardDamagePerTurn", 14)

        snakes = {}
        for s in game_state["board"]["snakes"]:
            body = [(p["x"], p["y"]) for p in s["body"]]
            snakes[s["id"]] = SnakeState(
                sid=s["id"],
                body=body,
                health=s["health"],
                alive=True,
            )

        food = {(f["x"], f["y"]) for f in game_state["board"].get("food", [])}
        hazards = {(h["x"], h["y"]) for h in game_state["board"].get("hazards", [])}
        return State(width, height, snakes, food, hazards, you_id, turn, hazard_damage)

    def clone(self):
        return State(
            self.width,
            self.height,
            {sid: snake.clone() for sid, snake in self.snakes.items()},
            self.food.copy(),
            self.hazards.copy(),
            self.you_id,
            self.turn,
            self.hazard_damage,
        )

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def alive_snakes(self):
        return [s for s in self.snakes.values() if s.alive]

    def is_terminal(self):
        me = self.snakes[self.you_id]
        return (not me.alive) or len(self.alive_snakes()) <= 1 or self.turn >= 300

    def occupied_without_tails(self):
        occ = set()
        for snake in self.alive_snakes():
            if len(snake.body) <= 1:
                occ.update(snake.body)
            else:
                occ.update(snake.body[:-1])
        return occ

    def legal_moves(self, sid):
        snake = self.snakes[sid]
        if not snake.alive:
            return []

        occupied = self.occupied_without_tails()
        moves = []
        for move_name in DIRS:
            nxt = add_pos(snake.head, move_name)
            if not self.in_bounds(nxt):
                continue
            if nxt in occupied:
                continue
            moves.append(move_name)
        return moves

    def local_liberties(self, pos, sid):
        occupied = self.occupied_without_tails()
        count = 0
        for move_name in DIRS:
            nxt = add_pos(pos, move_name)
            if self.in_bounds(nxt) and nxt not in occupied:
                count += 1
        return count

    def reachable_area(self, start, sid, max_cells=32):
        occupied = self.occupied_without_tails()
        if start in occupied and start != self.snakes[sid].head:
            return 0

        q = deque([start])
        seen = {start}
        area = 0

        while q and area < max_cells:
            cur = q.popleft()
            area += 1
            for move_name in DIRS:
                nxt = add_pos(cur, move_name)
                if not self.in_bounds(nxt):
                    continue
                if nxt in occupied and nxt != self.snakes[sid].head:
                    continue
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        return area

    def head_to_head_risk(self, sid, target):
        me = self.snakes[sid]
        risk = 0.0
        for other in self.alive_snakes():
            if other.sid == sid:
                continue
            if manhattan(other.head, target) == 1:
                if other.length >= me.length:
                    risk += 1.0
                else:
                    risk -= 0.25
        return risk

    def action_scores(self, sid):
        snake = self.snakes[sid]
        legal = self.legal_moves(sid)
        if not legal:
            return {}

        scores = {}
        center = ((self.width - 1) / 2.0, (self.height - 1) / 2.0)

        for move_name in legal:
            nxt = add_pos(snake.head, move_name)
            score = 0.0

            liberties = self.local_liberties(nxt, sid)
            score += 2.5 * liberties

            area = self.reachable_area(nxt, sid, max_cells=28)
            score += 0.10 * area

            food_dist = min((manhattan(nxt, f) for f in self.food), default=8)

            if nxt in self.food:
                score += 8.0 + max(0, (60 - snake.health) / 10.0)
            elif snake.health < 40:
                score += max(0, 8 - food_dist) * 0.9
            else:
                score += max(0, 6 - food_dist) * 0.15

            if nxt in self.hazards:
                score -= 8.0 + 0.15 * self.hazard_damage
                if snake.health < 30:
                    score -= 15.0

            score -= 2.5 * self.head_to_head_risk(sid, nxt)
            score -= 0.05 * (abs(nxt[0] - center[0]) + abs(nxt[1] - center[1]))

            scores[move_name] = score

        return scores

    def policy_move(self, sid, epsilon=0.15):
        legal = self.legal_moves(sid)
        if not legal:
            return random.choice(list(DIRS.keys()))

        if random.random() < epsilon:
            return random.choice(legal)

        scores = self.action_scores(sid)
        if not scores:
            return random.choice(legal)

        probs = softmax_dict(scores, temperature=1.8)
        r = random.random()
        c = 0.0
        for move_name, p in probs.items():
            c += p
            if r <= c:
                return move_name
        return max(scores, key=scores.get)

    def empty_cells(self):
        occupied = set()
        for snake in self.alive_snakes():
            occupied.update(snake.body)
        empties = []
        for x in range(self.width):
            for y in range(self.height):
                p = (x, y)
                if p in occupied or p in self.food:
                    continue
                empties.append(p)
        return empties

    def spawn_food_if_needed(self):
        empties = self.empty_cells()
        while len(self.food) < 2 and empties:
            cell = random.choice(empties)
            self.food.add(cell)
            empties.remove(cell)

    def step(self, actions):
        nxt_state = self.clone()
        nxt_state.turn += 1

        alive_ids = [s.sid for s in nxt_state.alive_snakes()]
        if not alive_ids:
            return nxt_state

        for sid in alive_ids:
            if actions.get(sid) not in DIRS:
                legal = nxt_state.legal_moves(sid)
                actions[sid] = legal[0] if legal else random.choice(list(DIRS.keys()))

        new_heads = {}
        new_bodies = {}
        new_health = {}
        grows = {}
        dead = set()

        for sid in alive_ids:
            snake = nxt_state.snakes[sid]
            nh = add_pos(snake.head, actions[sid])
            new_heads[sid] = nh

            if not nxt_state.in_bounds(nh):
                dead.add(sid)

            ate_food = nh in nxt_state.food
            grows[sid] = ate_food

            health = snake.health - 1
            if nh in nxt_state.hazards:
                health -= nxt_state.hazard_damage
            new_health[sid] = health
            if health <= 0:
                dead.add(sid)

            if ate_food:
                new_bodies[sid] = [nh] + snake.body[:]
            else:
                new_bodies[sid] = [nh] + snake.body[:-1]

        pos_to_ids = defaultdict(list)
        for sid in alive_ids:
            if sid not in dead:
                pos_to_ids[new_heads[sid]].append(sid)

        for _, ids in pos_to_ids.items():
            if len(ids) > 1:
                max_len = max(len(new_bodies[sid]) for sid in ids)
                winners = [sid for sid in ids if len(new_bodies[sid]) == max_len]
                if len(winners) != 1:
                    dead.update(ids)
                else:
                    winner = winners[0]
                    for sid in ids:
                        if sid != winner:
                            dead.add(sid)

        occupied_body = set()
        for sid in alive_ids:
            if sid in dead:
                continue
            for cell in new_bodies[sid][1:]:
                occupied_body.add(cell)

        for sid in alive_ids:
            if sid in dead:
                continue
            if new_heads[sid] in occupied_body:
                dead.add(sid)

        eaten = set()
        for sid in alive_ids:
            snake = nxt_state.snakes[sid]
            if sid in dead:
                snake.alive = False
                snake.body = []
                snake.health = 0
            else:
                snake.body = new_bodies[sid]
                if grows[sid]:
                    snake.health = 100
                    eaten.add(new_heads[sid])
                else:
                    snake.health = new_health[sid]

        nxt_state.food.difference_update(eaten)
        nxt_state.spawn_food_if_needed()
        return nxt_state

    def evaluate(self, sid):
        me = self.snakes[sid]
        if not me.alive:
            return 0.0

        alive = self.alive_snakes()
        if len(alive) == 1 and alive[0].sid == sid:
            return 1.0

        opp_lengths = [s.length for s in alive if s.sid != sid]
        avg_opp_len = sum(opp_lengths) / len(opp_lengths) if opp_lengths else me.length

        legal_count = len(self.legal_moves(sid))
        area = self.reachable_area(me.head, sid, max_cells=40)
        food_dist = min((manhattan(me.head, f) for f in self.food), default=8)

        survival_raw = 0.0
        survival_raw += 1.8 * legal_count
        survival_raw += 0.12 * area
        survival_raw += 0.03 * me.health
        survival_raw -= 2.0 if me.head in self.hazards else 0.0
        if me.health < 35:
            survival_raw += 0.8 * max(0, 8 - food_dist)

        for other in alive:
            if other.sid == sid:
                continue
            if manhattan(other.head, me.head) <= 2 and other.length >= me.length:
                survival_raw -= 1.5

        length_raw = me.length - avg_opp_len

        survival_score = 0.5 + 0.5 * math.tanh(survival_raw / 10.0)
        length_score = 0.5 + 0.5 * math.tanh(length_raw / 3.0)

        return 0.8 * survival_score + 0.2 * length_score


class Node:
    def __init__(self, state, my_id, parent=None, action=None):
        self.state = state
        self.my_id = my_id
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

        scores = state.action_scores(my_id) if (state.snakes[my_id].alive and not state.is_terminal()) else {}
        self.priors = softmax_dict(scores, temperature=1.5)
        self.unexpanded = sorted(scores.keys(), key=lambda a: self.priors[a], reverse=True)

    @property
    def q(self):
        if self.visits == 0:
            return 0.5
        return self.value_sum / self.visits


def sample_joint_actions(state, my_id, forced_my_action=None):
    actions = {}
    for snake in state.alive_snakes():
        sid = snake.sid
        if sid == my_id and forced_my_action is not None:
            actions[sid] = forced_my_action
        else:
            actions[sid] = state.policy_move(sid)
    return actions


def select_child(node, c_puct=C_PUCT):
    best_score = -1e18
    best_child = None
    parent_scale = math.sqrt(node.visits + 1)

    for action, child in node.children.items():
        prior = node.priors.get(action, 0.25)
        u = c_puct * prior * parent_scale / (1 + child.visits)
        score = child.q + u
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def rollout(state, my_id, depth_limit=ROLLOUT_DEPTH):
    cur = state.clone()
    for _ in range(depth_limit):
        if cur.is_terminal():
            break
        actions = sample_joint_actions(cur, my_id)
        cur = cur.step(actions)
    return cur.evaluate(my_id)


def choose_heuristic_move(state, my_id):
    scores = state.action_scores(my_id)
    if not scores:
        return random.choice(list(DIRS.keys()))
    return max(scores, key=scores.get)


def mcts_search(root_state, my_id, time_budget=TIME_BUDGET, rollout_depth=ROLLOUT_DEPTH):
    root = Node(root_state, my_id)

    legal = list(root.priors.keys())
    if not legal:
        return None, {}, 0
    if len(legal) == 1:
        return legal[0], {legal[0]: (1, 1.0)}, 1

    deadline = time.perf_counter() + time_budget
    iterations = 0

    while time.perf_counter() < deadline:
        node = root
        depth = 0

        while (not node.state.is_terminal()) and (not node.unexpanded) and node.children and depth < 16:
            node = select_child(node)
            depth += 1

        if (not node.state.is_terminal()) and node.unexpanded:
            action = node.unexpanded.pop(0)
            child_state = node.state.step(sample_joint_actions(node.state, my_id, forced_my_action=action))
            child = Node(child_state, my_id, parent=node, action=action)
            node.children[action] = child
            node = child

        value = rollout(node.state, my_id, depth_limit=rollout_depth)

        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

        iterations += 1

    stats = {}
    best_action = None
    best_visits = -1
    best_q = -1.0

    for action in legal:
        if action in root.children:
            child = root.children[action]
            q = child.q
            v = child.visits
        else:
            q = 0.0
            v = 0
        stats[action] = (v, q)
        if v > best_visits or (v == best_visits and q > best_q):
            best_action = action
            best_visits = v
            best_q = q

    if best_action is None:
        best_action = max(root.priors, key=root.priors.get)
    return best_action, stats, iterations


def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "YOUR_NAME",
        "color": "#2D6A4F",
        "head": "default",
        "tail": "default",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\\n")


def move(game_state: typing.Dict) -> typing.Dict:
    state = State.from_game_state(game_state)
    my_id = state.you_id

    legal = state.legal_moves(my_id)
    if not legal:
        print(f"MOVE {game_state['turn']}: no legal moves, fallback down")
        return {"move": "down"}

    if len(legal) == 1:
        print(f"MOVE {game_state['turn']}: forced {legal[0]}")
        return {"move": legal[0]}

    next_move, stats, iterations = mcts_search(state, my_id)

    if next_move is None:
        next_move = choose_heuristic_move(state, my_id)

    print(f"MOVE {game_state['turn']}: {next_move} | iters={iterations} | stats={stats}")
    return {"move": next_move}


if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
