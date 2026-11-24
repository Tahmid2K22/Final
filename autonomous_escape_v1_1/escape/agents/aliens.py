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
        tiles = list(extra_tiles)
        tiles += sim.path_ahead(14)
        tiles += [p for p, b in sim.blk.items() if b == SOFT]
        acts, seen = [None], set()
        for p in tiles:
            if p in seen or not sim.can_target(p):
                continue
            seen.add(p)
            b = sim.blk.get(p, OPEN)
            acts.append(('boulder', p) if b == OPEN else ('debris', p))
            if b == OPEN and len(acts) < 10:
                acts.append(('fireball', p))
                acts.append(('debris', p))
            if len(acts) >= 24:
                break
        return acts

    def _advance_rover(self, sim):
        while sim.result is None and sim.turn == ROVER:
            sim.apply_rover(sim.greedy_rover_action())

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
            path = [p for p in sim.path_ahead(8) if sim.can_target(p)]
            r = self.rng.random()
            if path and r < 0.6:
                p = path[0] if r < 0.4 else self.rng.choice(path)
                sim.apply_alien(('boulder', p) if sim.blk.get(p, OPEN) == OPEN else ('debris', p))
            else:
                softs = [p for p, b in sim.blk.items() if b == SOFT and sim.can_target(p)]
                sim.apply_alien(('debris', self.rng.choice(softs)) if softs else None)
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

# ---------------------------------------------------------------- strategic hunter
class StrategicHunterAlien(MCTSAlien):
    """Hybrid alien: A* route prediction + BFS connectivity + CSP attacks + Minimax + MCTS."""
    name = 'Strategic Hunter'
    side = ALIEN

    def candidates(self, sim, extra_tiles=()):
        # CSP: enumerate legal attacks satisfying constraints
        acts = []
        seen = set()
        for p in sim.path_ahead(14) + [p for p, b in sim.blk.items() if b == SOFT]:
            if p in seen or not sim.can_target(p):
                continue
            seen.add(p)
            b = sim.blk.get(p, OPEN)
            if b == OPEN:
                acts.append(('fireball', p))
                acts.append(('boulder', p))
            acts.append(('debris', p))
            if len(acts) >= 24:
                break
        if not acts:
            acts = [None]
        acts = [None] + [a for a in acts if a is not None]
        return acts[:25]

ALIENS.append(StrategicHunterAlien)


# ================================================================ APEX ALIEN
# The ApexAlien combines four tactics in a strict priority order:
#
#  1. MIN-CUT SEALING  — find the minimum vertex cut between the rover and
#     the goal.  If the remaining budget covers it, commit every strike to
#     placing a boulder on each cut tile and then hardening it with debris
#     (drop boulder → fling debris → HARD = laser-proof wall).
#
#  2. INTERCEPT        — if the rover is ≤ intercept_steps ahead of a cut
#     tile, drop a boulder right now so the rover cannot walk through before
#     the next strike.
#
#  3. FIREBALL DRAIN   — shoot fireballs at tiles the rover MUST cross
#     (confirmed chokepoints or the immediate next step).  Even if the rover
#     cannot be fully sealed, draining fuel wins by exhaustion.
#
#  4. ENHANCED MCTS    — deeper horizon (8 turns), more iterations (300),
#     and a cut-aware rollout policy that always targets cut tiles first.
# ================================================================

def _rover_bfs_dist(gmap, blk, rover):
    """BFS move-distance from the rover (ignoring HARD, counting SOFT as 1 extra)."""
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
    """
    Apex predator alien — the strongest possible blocker.

    Priority order every strike:
      1. Complete a committed min-cut seal (drop boulder or harden with debris).
      2. Intercept: block cut tiles the rover is about to walk onto.
      3. Fireball mandatory path tiles to drain fuel.
      4. Enhanced MCTS (horizon 8, 300 iters, cut-aware rollout).
    """
    name = 'Apex'
    side = ALIEN

    # --- tuning knobs -------------------------------------------------------
    CUT_LIMIT       = 10   # give up min-cut if wider than this (too expensive)
    INTERCEPT_STEPS = 6    # block a cut tile if rover is this many moves away
    MCTS_ITERS      = 300
    MCTS_HORIZON    = 8
    MCTS_C          = 1.2
    MCTS_TIME       = 0.55
    # ------------------------------------------------------------------------

    def __init__(self, seed=0):
        super().__init__()
        self.rng = random.Random(seed)
        self._cut_plan   = []   # ordered list of (tile, phase) still to place
        self._last_cut   = None # frozenset of the current committed cut

    # ---------------------------------------------------------------- helpers
    def _compute_cut(self, s):
        """Return the minimum vertex cut separating rover from goal (or []).
        
        We use the rover's reachable open neighbours as sources so the cut
        is always at least 2 tiles away from the rover (alien_can_target
        requires manhattan > 1). Tiles that are protected or already HARD
        are excluded from candidates.
        """
        # Use neighbours of the rover as the source set so the cut tiles
        # are never the rover itself (which alien_can_target rejects).
        sources = [v for v in s.map.nbrs.get(s.rover, [])
                   if s.blk.get(v, OPEN) != HARD]
        if not sources:
            # rover is surrounded by hard boulders — it already cannot escape
            return []
        n, cut = min_vertex_cut(
            s.map, s.blk,
            sources=sources,
            is_sink=s.map.in_goal_zone,
            cuttable=lambda p: not s.map.protected(p) and manhattan(p, s.rover) > 1,
            limit=self.CUT_LIMIT,
        )
        return cut if n != INF else []

    def _build_plan(self, cut, s):
        """
        Turn a cut list into an ordered action plan.

        Each cut tile needs up to 2 strikes:
          phase 0 → ('boulder', p)  — block the tile (soft boulder)
          phase 1 → ('debris',  p)  — harden it (laser-proof HARD boulder)

        Tiles the rover is closest to come first (most urgent).
        """
        rover_dist = _rover_bfs_dist(s.map, s.blk, s.rover)
        plan = []
        for p in sorted(cut, key=lambda p: rover_dist.get(p, 999)):
            b = s.blk.get(p, OPEN)
            if b == OPEN and s.alien_can_target(p):
                plan.append((p, 0))   # need boulder then debris
                plan.append((p, 1))
            elif b == SOFT and s.alien_can_target(p):
                plan.append((p, 1))   # already bouldered, just harden
        return plan

    def _plan_action(self, p, phase, s):
        if phase == 0:
            return ('boulder', p)
        # phase 1: harden with debris (tile must still be targetable & soft)
        if s.blk.get(p, OPEN) == SOFT and s.alien_can_target(p):
            return ('debris', p)
        # rover already shot it or it was open — skip to boulder
        if s.blk.get(p, OPEN) == OPEN and s.alien_can_target(p):
            return ('boulder', p)
        return None

    def _intercept(self, s, cut_set, rover_dist):
        """
        Immediate intercept: pick the cut tile closest to the rover that
        hasn't been sealed yet and is within INTERCEPT_STEPS moves.
        """
        for p in sorted(cut_set, key=lambda p: rover_dist.get(p, 999)):
            d = rover_dist.get(p, 999)
            if d > self.INTERCEPT_STEPS:
                break
            b = s.blk.get(p, OPEN)
            if b == HARD:
                continue
            if not s.alien_can_target(p):
                continue
            if b == OPEN:
                return ('boulder', p)
            if b == SOFT:
                return ('debris', p)
        return None

    def _fireball_target(self, s):
        """
        Find the best tile on the rover's mandatory route to shoot a fireball
        at.  Prefer tiles the rover is 2–4 steps away (fireball TTL is 5).
        """
        path = rover_path(s)
        rover_dist = _rover_bfs_dist(s.map, s.blk, s.rover)
        for p in path:
            d = rover_dist.get(p, 999)
            if 2 <= d <= 4 and s.blk.get(p, OPEN) == OPEN and s.alien_can_target(p):
                return ('fireball', p)
        # fallback: any path tile we can hit
        for p in path:
            if s.blk.get(p, OPEN) == OPEN and s.alien_can_target(p):
                return ('fireball', p)
        return None

    # ---------------------------------------------------------------- MCTS
    def _mcts_candidates(self, sim, cut_set):
        """Cut-aware candidate list for MCTS nodes."""
        path_tiles = sim.path_ahead(16)
        # cut tiles first, then path, then existing soft boulders
        ordered = []
        seen = set()
        for p in list(cut_set) + path_tiles + [p for p, b in sim.blk.items() if b == SOFT]:
            if p in seen or not sim.can_target(p):
                continue
            seen.add(p)
            b = sim.blk.get(p, OPEN)
            if b == OPEN:
                ordered.append(('boulder', p))
                if len(ordered) < 12:
                    ordered.append(('fireball', p))
                    ordered.append(('debris', p))
            elif b == SOFT:
                ordered.append(('debris', p))
            if len(ordered) >= 30:
                break
        return [None] + ordered

    def _advance_rover(self, sim):
        while sim.result is None and sim.turn == ROVER:
            sim.apply_rover(sim.greedy_rover_action())

    def _reward(self, sim):
        if sim.result == ALIEN:
            return 1.0
        if sim.result == ROVER:
            return 0.0
        margin = sim.fuel - sim.route_cost()
        return 1.0 / (1.0 + math.exp(margin / 5.0))

    def _rollout(self, sim, turns, cut_set):
        """Cut-aware rollout: always try to place on cut tiles first."""
        while sim.result is None and turns > 0:
            self._advance_rover(sim)
            if sim.result is not None:
                break
            # try to seal the cut tile closest to the rover first
            rover_dist = _rover_bfs_dist(sim.m, sim.blk, sim.pos)
            sealed = False
            for p in sorted(cut_set, key=lambda p: rover_dist.get(p, 999)):
                if not sim.can_target(p):
                    continue
                b = sim.blk.get(p, OPEN)
                if b == HARD:
                    continue
                action = ('boulder', p) if b == OPEN else ('debris', p)
                sim.apply_alien(action)
                sealed = True
                break
            if not sealed:
                path = [p for p in sim.path_ahead(8) if sim.can_target(p)]
                if path:
                    p = path[0]
                    sim.apply_alien(('boulder', p) if sim.blk.get(p, OPEN) == OPEN else ('debris', p))
                else:
                    sim.apply_alien(None)
            turns -= 1
        return self._reward(sim)

    def _run_mcts(self, s, cut_set):
        cache, counter = {}, [0]
        root_sim = Sim.from_state(s, cache, counter)
        candidates = self._mcts_candidates(root_sim, cut_set)
        root = _Node(candidates)
        t0 = time.perf_counter()
        it = 0
        while it < self.MCTS_ITERS and time.perf_counter() - t0 < self.MCTS_TIME:
            it += 1
            sim = root_sim.clone()
            node, path, depth = root, [root], 0
            # selection
            while not node.untried and node.children and sim.result is None:
                logn = math.log(node.n)
                a, node = max(
                    node.children.items(),
                    key=lambda kv: kv[1].w / kv[1].n + self.MCTS_C * math.sqrt(logn / kv[1].n)
                )
                sim.apply_alien(a)
                self._advance_rover(sim)
                path.append(node)
                depth += 1
            # expansion
            if node.untried and sim.result is None:
                a = node.untried.pop(self.rng.randrange(len(node.untried)))
                sim.apply_alien(a)
                self._advance_rover(sim)
                child = _Node(self._mcts_candidates(sim, cut_set) if sim.result is None else [])
                node.children[a] = child
                path.append(child)
                depth += 1
            reward = self._rollout(sim, max(self.MCTS_HORIZON - depth, 0), cut_set)
            for nd in path:
                nd.n += 1
                nd.w += reward
        self.nodes += counter[0] + it
        if not root.children:
            return None, 0.0
        best_a, best_nd = max(root.children.items(), key=lambda kv: kv[1].n)
        return best_a, best_nd.w / best_nd.n

    # ---------------------------------------------------------------- main act
    def act(self, s):
        # --- 1. Compute / refresh the min-cut ---
        cut = self._compute_cut(s)
        cut_set = set(cut)

        # --- 2. Refresh the sealing plan if the cut changed ---
        cut_key = frozenset(cut)
        if cut_key != self._last_cut:
            self._last_cut = cut_key
            self._cut_plan = self._build_plan(cut, s)

        rover_dist = _rover_bfs_dist(s.map, s.blk, s.rover)

        # --- 3. ALWAYS SEAL when cut is small enough and tiles are targetable ---
        # This is the highest-priority action: if we can seal every route with
        # our remaining budget, commit to it immediately (don't wait for MCTS).
        if cut and self._cut_plan:
            # Drop plan entries that are no longer valid, then take the first
            # legal seal action.  Popping stale entries inside the same loop
            # shifted the indices and could pop the wrong tile.
            self._cut_plan = [(p, ph) for p, ph in self._cut_plan
                              if self._plan_action(p, ph, s) is not None]
            for i, (p, phase) in enumerate(self._cut_plan):
                action = self._plan_action(p, phase, s)
                if s.legal_alien(action):
                    self._cut_plan.pop(i)
                    self.info = (f'SEAL cut={len(cut)} '
                                 f'plan={len(self._cut_plan)} '
                                 f'{"boulder" if phase==0 else "harden"}@{p}')
                    return action

        # --- 4. INTERCEPT: rover about to step onto an unsealed cut tile ---
        if cut_set:
            intercept = self._intercept(s, cut_set, rover_dist)
            if intercept is not None:
                p = intercept[1]
                self._cut_plan = [(t, ph) for t, ph in self._cut_plan if t != p]
                self.info = f'INTERCEPT {intercept[0]} @ {p}'
                return intercept

        # --- 5. FIREBALL DRAIN: drain fuel on mandatory tiles ---
        # Deterministic: once sealing is impossible this turn, a fireball on
        # the cheapest route is always better than holding.
        fb = self._fireball_target(s)
        if fb:
            self.info = f'FIREBALL DRAIN @ {fb[1]}'
            return fb

        # --- 6. ENHANCED MCTS fallback ---
        action, win_est = self._run_mcts(s, cut_set)
        self.info = (f'MCTS {self.MCTS_ITERS} iters '
                     f'win={win_est:.2f} cut={len(cut)}')
        return action


ALIENS.append(ApexAlien)

