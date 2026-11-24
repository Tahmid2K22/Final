"""Alien agents. They see the whole GameState and must not modify it."""
import math
import random
import time

from ..config import OPEN, SOFT, HARD, INF
from ..graph import astar, bfs_moves, min_vertex_cut, manhattan
from ..fastsim import Sim, ROVER, ALIEN
from .base import Agent


def action_for(state, p):
    """Boulder on an open tile, debris on a soft one (to harden it)."""
    return ('boulder', p) if state.blk.get(p, OPEN) == OPEN else ('debris', p)


def rover_path(state):
    _, path = astar(state.map, state.blk, state.rover, state.map.goal, state.rules)
    return path[1:]


class RandomAlien(Agent):
    """Dormant tier: any legal tile, no plan."""
    name = 'Random'
    side = ALIEN

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def act(self, s):
        tiles = [p for p in s.map.open_tiles if s.alien_can_target(p)]
        return action_for(s, self.rng.choice(tiles)) if tiles else None


class GreedyAlien(Agent):
    """Reactive tier: block (or harden) the next legal tile on the rover's
    cheapest route."""
    name = 'Greedy'
    side = ALIEN

    def act(self, s):
        """Block one of the first few legal tiles on the rover's route, picked at
        random (nearer ones more likely)."""
        options = [p for p in rover_path(s) if s.alien_can_target(p)][:3]
        if not options:
            return None
        rng = getattr(self, 'rng', None)
        p = options[0] if rng is None else rng.choices(options, weights=[3, 2, 1][:len(options)])[0]
        if s.blk.get(p, OPEN) == OPEN and (rng is None or rng.random() < 0.5):
            return ('fireball', p)
        return action_for(s, p)


class _Node:
    __slots__ = ('children', 'untried', 'n', 'w')

    def __init__(self, actions):
        self.children = {}
        self.untried = list(actions)
        self.n = 0
        self.w = 0.0


class MCTSAlien(Agent):
    """UCT over the alien's decisions. The rover is modelled as a greedy
    cost-to-go follower on the true map; rollouts use a cheap block-the-route
    policy and are cut off after `horizon` alien turns, scored by how far the
    rover's route cost exceeds its fuel."""
    name = 'MCTS'
    side = ALIEN

    def __init__(self, iterations=150, time_limit=0.6, horizon=4, c=0.7, seed=0):
        super().__init__()
        self.iterations, self.time_limit = iterations, time_limit
        self.horizon, self.c = horizon, c
        self.rng = random.Random(seed)

    def candidates(self, sim, extra_tiles=()):
        path = sim.path_ahead(14)
        path_set = set(path)
        goal_d = sim.dist()
        pos_d = goal_d.get(sim.pos, INF)
        # Only consider existing soft boulders that are actually ahead of the rover
        ahead_softs = [p for p, b in sim.blk.items()
                       if b == SOFT and (p in path_set or goal_d.get(p, INF) < pos_d)]
        tiles = list(extra_tiles) + path + ahead_softs
        acts, seen = [None], set()
        for p in tiles:
            if p in seen or not sim.can_target(p):
                continue
            seen.add(p)
            b = sim.blk.get(p, OPEN)
            acts.append(('boulder', p) if b == OPEN else ('debris', p))
            if b == OPEN:
                # Only use fireballs within TTL reach of the rover
                if manhattan(p, sim.pos) <= min(4, sim.r.fireball_ttl - 1) and len(acts) < 12:
                    acts.append(('fireball', p))
                if len(acts) < 10:
                    acts.append(('debris', p))
            if len(acts) >= 24:
                break
        return acts

    def _advance_rover(self, sim):
        while sim.result is None and sim.turn == ROVER:
            sim.apply_rover(_smart_rover_action(sim))

    def _reward(self, sim):
        if sim.result == ALIEN:
            return 1.0
        if sim.result == ROVER:
            return 0.0
        margin = sim.fuel - sim.route_cost()
        return 1.0 / (1.0 + math.exp(margin / 4.0))

    def _rollout(self, sim, turns):
        while sim.result is None and turns > 0:
            self._advance_rover(sim)
            if sim.result is not None:
                break
            path = [p for p in sim.path_ahead(10) if sim.can_target(p)]
            # Prefer hardening a soft boulder on the active path, otherwise block ahead
            soft_on_path = [p for p in path if sim.blk.get(p, OPEN) == SOFT]
            if soft_on_path and self.rng.random() < 0.7:
                sim.apply_alien(('debris', soft_on_path[0]))
            elif path:
                r = self.rng.random()
                p = path[0] if r < 0.55 else self.rng.choice(path[:5])
                sim.apply_alien(('boulder', p) if sim.blk.get(p, OPEN) == OPEN else ('debris', p))
            else:
                sim.apply_alien(None)
            turns -= 1
        return self._reward(sim)

    def act(self, s):
        cache, counter = {}, [0]
        root_sim = Sim.from_state(s, cache, counter)
        root = _Node(self.candidates(root_sim, ()))
        t0 = time.perf_counter()
        it = 0
        while it < self.iterations and time.perf_counter() - t0 < self.time_limit:
            it += 1
            sim = root_sim.clone()
            node, path, depth = root, [root], 0
            # selection
            while not node.untried and node.children and sim.result is None:
                logn = math.log(node.n)
                a, node = max(node.children.items(), key=lambda kv: kv[1].w / kv[1].n
                              + self.c * math.sqrt(logn / kv[1].n))
                sim.apply_alien(a)
                self._advance_rover(sim)
                path.append(node)
                depth += 1
            # expansion
            if node.untried and sim.result is None:
                a = node.untried.pop(self.rng.randrange(len(node.untried)))
                sim.apply_alien(a)
                self._advance_rover(sim)
                child = _Node(self.candidates(sim) if sim.result is None else [])
                node.children[a] = child
                path.append(child)
                depth += 1
            reward = self._rollout(sim, max(self.horizon - depth, 0))
            for n in path:
                n.n += 1
                n.w += reward
        self.nodes += counter[0] + it
        if not root.children:
            return None
        best_a, best = max(root.children.items(), key=lambda kv: kv[1].n)
        self.info = f'{it} rollouts, best win est. {best.w / best.n:.2f}'
        return best_a


ALIENS = [RandomAlien, GreedyAlien, MCTSAlien]

# ---------------------------------------------------------------- helpers for smart aliens
def _effective_blk(s, rover_dist):
    """Block map treating SOFT boulders with guaranteed-safe incoming debris
    as already HARD (since debris will land before the rover can shoot them)."""
    eff = dict(s.blk)
    for p, ttl in s.pending.items():
        if eff.get(p, OPEN) == SOFT and rover_dist.get(p, 999) > ttl:
            eff[p] = HARD
    return eff


def _smart_rover_action(sim):
    """Realistic rover policy for lookahead: accounts for fireballs and
    imminent debris (like Minimax/Fuzzy/EI), rather than walking blindly."""
    d = sim.dist()
    best, best_v = ('wait', None), INF
    if sim.pending.get(sim.pos) != 1:
        best_v = sim.r.move_cost + d.get(sim.pos, INF) + 0.5
    for v in sim.m.nbrs[sim.pos]:
        b = sim.blk.get(v, OPEN)
        if b == HARD:
            continue
        val = sim.r.enter_cost(b) + d.get(v, INF)
        if v in sim.fireballs:
            val += sim.r.fireball_damage
        if sim.pending.get(v) == 1:
            val += sim.r.debris_damage
        if val < best_v:
            best_v = val
            best = ('shoot', v) if b == SOFT else ('move', v)
    return best


# ---------------------------------------------------------------- strategic hunter
class StrategicHunterAlien(Agent):
    """Hybrid alien: A* route prediction + BFS connectivity + min-vertex cut
    along the rover's active path + lookahead fuel-drain harassment."""
    name = 'Strategic Hunter'
    side = ALIEN

    def __init__(self, seed=0):
        super().__init__()
        self.rng = random.Random(seed)

    def act(self, s):
        apex = getattr(self, '_core', None)
        if apex is None:
            apex = ApexAlien(seed=0)
            apex.rng = self.rng
            self._core = apex
        a = apex.act(s)
        self.nodes = apex.nodes
        self.info = apex.info
        return a


ALIENS.append(StrategicHunterAlien)


# ================================================================ APEX ALIEN
# Game-theoretically grounded alien AI:
#
#  1. INSTANT WIN      — drop a boulder or harden a soft boulder on the rover's
#     path if doing so immediately makes route_cost > fuel or INF.
#
#  2. HARDEN ON-PATH   — never leave a placed SOFT boulder on the active cut
#     or rover's route unhardened while wandering off-path; immediately seal it
#     with debris if the rover cannot reach and shoot it before landing.
#
#  3. PATH-FIRST CUT   — compute the minimum vertex cut that respects lead time
#     (so OPEN cut tiles are far enough ahead to survive until hardened), and
#     ALWAYS strike the cut tile ON the rover's current optimal path first!
#     By Max-Flow Min-Cut, the rover's path MUST intersect the cut, so ~100%
#     of strikes hit where the rover is actually heading and force early
#     detours/backtracking.
#
#  4. PATH BOTTLENECK & LOOKAHEAD DRAIN — when the cut is too wide to seal
#     within the remaining ticks, evaluate tiles strictly on the rover's active
#     route (and belief route): prioritize permanent HARD chokepoints with huge
#     detour gain, or verify boulder/debris/fireball with a smart-rover Sim.
# ================================================================

def _rover_bfs_dist(gmap, blk, rover):
    """BFS move-distance from the rover (ignoring HARD)."""
    from collections import deque
    dist = {rover: 0}
    dq = deque([rover])
    while dq:
        u = dq.popleft()
        for v in gmap.nbrs[u]:
            if v not in dist and blk.get(v, OPEN) != HARD:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


class ApexAlien(Agent):
    """Path-first min-cut sealer and bottleneck hunter."""
    name = 'Apex'
    side = ALIEN

    CUT_LIMIT = 10

    def __init__(self, seed=0):
        super().__init__()
        self.rng = random.Random(seed)

    def _detour_gain(self, s, base_blk, base_cost, p, becomes=HARD, counter=None):
        test_blk = dict(base_blk)
        test_blk[p] = becomes
        nc, _ = astar(s.map, test_blk, s.rover, s.map.goal, s.rules, counter=counter)
        if nc == INF:
            return INF
        return nc - base_cost

    def _compute_cut(self, s, eff_blk, rover_dist):
        """Find a sealable min-vertex-cut. Prefer a forward-safe cut where OPEN
        tiles are far enough ahead of the rover to be hardened in 2 strikes;
        fall back to the unrestricted cut if needed."""
        sources = [v for v in s.map.nbrs.get(s.rover, [])
                   if eff_blk.get(v, OPEN) != HARD]
        if not sources:
            return []
        safe_d = min(s.rules.alien_every + 1, 7)
        n_safe, cut_safe = min_vertex_cut(
            s.map, eff_blk,
            sources=sources,
            is_sink=s.map.in_goal_zone,
            cuttable=lambda p: (not s.map.protected(p) and manhattan(p, s.rover) > 1
                                and (eff_blk.get(p, OPEN) == SOFT or rover_dist.get(p, 999) >= safe_d)),
            limit=self.CUT_LIMIT,
        )
        n_full, cut_full = min_vertex_cut(
            s.map, eff_blk,
            sources=sources,
            is_sink=s.map.in_goal_zone,
            cuttable=lambda p: not s.map.protected(p) and manhattan(p, s.rover) > 1,
            limit=self.CUT_LIMIT,
        )
        if n_safe != INF and (n_full == INF or n_safe <= n_full + 1):
            return cut_safe
        return cut_full if n_full != INF else []

    def _simulate_candidate(self, s, action, cache, counter):
        """Look ahead over ~2 strike intervals against both a smart hazard-
        avoiding rover and a greedy path rover; return expected slack reduction."""
        horizon = min(max(8, 2 * s.rules.alien_every + 2), 18)
        scores = []
        for rover_fn in (_smart_rover_action, lambda sm: sm.greedy_rover_action()):
            sim = Sim.from_state(s, cache, counter)
            sim.apply_alien(action)
            if sim.result == ALIEN:
                scores.append(10000.0)
                continue
            if sim.result == ROVER:
                scores.append(-10000.0)
                continue
            follow_used = False
            for _ in range(horizon):
                if sim.result is not None:
                    break
                if sim.turn == ALIEN:
                    if not follow_used and action is not None and action[0] == 'boulder':
                        tp = action[1]
                        if sim.blk.get(tp, OPEN) == SOFT and sim.can_target(tp):
                            sim.apply_alien(('debris', tp))
                            follow_used = True
                            continue
                    nxt = next((q for q in sim.path_ahead(10) if sim.can_target(q)), None)
                    if nxt is not None:
                        b = sim.blk.get(nxt, OPEN)
                        sim.apply_alien(('boulder', nxt) if b == OPEN else ('debris', nxt))
                    else:
                        sim.apply_alien(None)
                else:
                    sim.apply_rover(rover_fn(sim))
            if sim.result == ALIEN:
                scores.append(10000.0)
            elif sim.result == ROVER:
                scores.append(-10000.0)
            else:
                scores.append(float(sim.route_cost() - sim.fuel))
        return 0.75 * scores[0] + 0.25 * scores[1]

    def act(self, s):
        counter = [0]
        rover_dist = _rover_bfs_dist(s.map, s.blk, s.rover)
        eff_blk = _effective_blk(s, rover_dist)

        raw_cost, raw_path = astar(s.map, s.blk, s.rover, s.map.goal, s.rules, counter=counter)
        eff_cost, eff_path = astar(s.map, eff_blk, s.rover, s.map.goal, s.rules, counter=counter)
        _, belief_path = astar(s.map, s.belief, s.rover, s.map.goal, s.rules, counter=counter)

        path_tiles = eff_path[1:] if len(eff_path) > 1 else raw_path[1:]
        path_pos = {p: i for i, p in enumerate(path_tiles, 1)}
        belief_set = set(belief_path[1:])
        raw_set = set(raw_path[1:])

        # --- 1. INSTANT WIN CHECK ---
        for p in raw_path[1:20]:
            if not s.alien_can_target(p):
                continue
            b = s.blk.get(p, OPEN)
            if b == OPEN:
                gain = self._detour_gain(s, s.blk, raw_cost, p, SOFT, counter)
                if raw_cost + gain > s.fuel:
                    self.nodes += counter[0]
                    self.info = f'WIN boulder @ {p}'
                    return ('boulder', p)
            elif b == SOFT and rover_dist.get(p, 999) >= 3:
                gain = self._detour_gain(s, s.blk, raw_cost, p, HARD, counter)
                if gain == INF or raw_cost + gain > s.fuel:
                    self.nodes += counter[0]
                    self.info = f'WIN harden @ {p}'
                    return ('debris', p)

        # --- 2. COMPUTE MIN-CUT & FEASIBILITY ---
        cut = self._compute_cut(s, eff_blk, rover_dist)
        cut_set = set(cut)
        need = sum(2 if eff_blk.get(p, OPEN) == OPEN else 1
                   for p in cut if eff_blk.get(p, OPEN) != HARD)
        est_strikes = (eff_cost if eff_cost != INF else raw_cost) / max(1, s.rules.alien_every)
        cut_feasible = bool(cut) and (need <= est_strikes * 1.35 + 2 or len(cut) <= 4)

        # --- 3. HARDEN ALREADY-PLACED SOFT BOULDERS ON CUT OR ACTIVE PATH ---
        soft_candidates = []
        for p, b in s.blk.items():
            if b != SOFT or not s.alien_can_target(p):
                continue
            d = rover_dist.get(p, 999)
            if d < 3:
                continue
            in_cut = cut_feasible and p in cut_set
            on_route = p in path_pos or p in raw_set or p in belief_set
            if not (in_cut or on_route):
                continue
            gain = self._detour_gain(s, eff_blk, eff_cost if eff_cost != INF else raw_cost, p, HARD, counter)
            can_finish_cut = in_cut and need <= est_strikes + 1 and d <= raw_cost
            if on_route or gain >= 4 or can_finish_cut:
                prio = (
                    1 if p in path_pos else 0,
                    1 if in_cut else 0,
                    1 if p in belief_set else 0,
                    gain if gain != INF else 9999,
                    -d,
                )
                soft_candidates.append((prio, p, gain))
        if soft_candidates:
            soft_candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_p, gain = soft_candidates[0]
            self.nodes += counter[0]
            self.info = f'HARDEN @ {best_p} (cut={len(cut)}, +cost={gain})'
            return ('debris', best_p)

        # --- 4. PATH-FIRST MIN-CUT SEALING ---
        if cut_feasible:
            cut_targets = [p for p in cut if s.alien_can_target(p) and eff_blk.get(p, OPEN) != HARD]
            if cut_targets:
                def cut_key(p):
                    on_eff = 1 if p in path_pos else 0
                    on_raw = 1 if p in raw_set else 0
                    on_bel = 1 if p in belief_set else 0
                    b = s.blk.get(p, OPEN)
                    d = rover_dist.get(p, 999)
                    safe = 1 if (b == SOFT or d >= min(s.rules.alien_every + 1, 7)) else 0
                    idx = path_pos.get(p, 9999)
                    return (on_eff, on_raw, on_bel, safe, 1 if b == SOFT else 0, -idx, -d)

                best_p = max(cut_targets, key=cut_key)
                act = ('debris', best_p) if s.blk.get(best_p, OPEN) == SOFT else ('boulder', best_p)
                self.nodes += counter[0]
                self.info = f'SEAL cut={len(cut)} need={need} {act[0]}@{best_p}'
                return act

        # --- 5. WIDE-CUT / OPEN-FIELD PATH BOTTLENECK & LOOKAHEAD DRAIN ---
        ordered_path = []
        seen = set()
        for p in path_tiles + [q for q in belief_path[1:] if q not in path_pos]:
            if p not in seen and s.alien_can_target(p):
                seen.add(p)
                ordered_path.append(p)

        safe_d = min(s.rules.alien_every + 1, 6)
        base_c = eff_cost if eff_cost != INF else raw_cost
        best_choke = None
        for idx, p in enumerate(ordered_path[:30]):
            d = rover_dist.get(p, 999)
            b = s.blk.get(p, OPEN)
            if (b == SOFT and d >= 3) or (b == OPEN and d >= safe_d):
                gain = self._detour_gain(s, eff_blk, base_c, p, HARD, counter)
                if gain >= 5:
                    score = (gain if gain != INF else 9999) - 0.15 * idx
                    if best_choke is None or score > best_choke[0]:
                        best_choke = (score, p, b, gain)

        if best_choke is not None:
            _, p, b, gain = best_choke
            act = ('debris', p) if b == SOFT else ('boulder', p)
            self.nodes += counter[0]
            self.info = f'CHOKE {act[0]}@{p} (+cost={gain})'
            return act

        cands = []
        for p in ordered_path[:8]:
            b = s.blk.get(p, OPEN)
            d = rover_dist.get(p, 999)
            if b == OPEN:
                cands.append(('boulder', p))
                if d <= 4:
                    cands.append(('fireball', p))
                if d <= 3:
                    cands.append(('debris', p))
            elif b == SOFT:
                cands.append(('debris', p))

        if not cands:
            self.nodes += counter[0]
            self.info = 'no legal target'
            return None

        cache = {}
        best_a = max(cands[:14], key=lambda a: self._simulate_candidate(s, a, cache, counter))
        self.nodes += counter[0]
        self.info = f'HUNT {best_a[0]}@{best_a[1]}'
        return best_a


ALIENS.append(ApexAlien)

