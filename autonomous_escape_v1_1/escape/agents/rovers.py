"""Rover agents. Each one sees only rover_view(): walls, its own boulder
memory, incoming debris, fuel and public counters."""
import numpy as np

from ..config import OPEN, SOFT, INF
from ..graph import astar, bfs_moves, manhattan
from ..fastsim import Sim, ROVER
from .base import Agent, path_action


class AStarRover(Agent):
    """R1 baseline: weighted A* on the belief map, replanned every turn.
    Soft boulders cost a shot plus a move; hard boulders are walls."""
    name = 'A*'
    side = ROVER

    def act(self, v):
        c = [0]
        _, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=self.tie_noise(v.map), counter=c)
        self.nodes += c[0]
        return path_action(v, path[1] if len(path) > 1 else None)


class MinimaxRover(Agent):
    """R3 anticipatory rover: alpha-beta over its own actions and the alien's
    next strike, on the belief map. Leaf value = fuel left after the cheapest
    remaining route (higher is better for the rover)."""
    name = 'Minimax'
    side = ROVER

    def __init__(self, depth=4, alien_branch=6):
        super().__init__()
        self.depth = depth
        self.alien_branch = alien_branch

    def act(self, v):
        cache, counter = {}, [0]
        root = Sim.from_view(v, cache, counter)
        best, best_val = ('wait', None), -INF
        alpha = -INF
        for a in self._ordered(root):
            child = root.clone()
            child.apply_rover(a)
            val = self._value(child, self.depth - 1, alpha, INF)
            if val > best_val:
                best, best_val = a, val
            alpha = max(alpha, val)
        self.nodes += counter[0]
        self.info = f'look-ahead value {best_val:.0f}'
        return best

    # Damage is loss progress, not fuel, so the leaf prices it in fuel units:
    # one fireball (10 damage) is worth roughly a short detour, and every point
    # of damage already taken makes the remaining route riskier.
    FIREBALL_RISK = 8.0
    DEBRIS_RISK = 5.0
    DAMAGE_RISK = 0.6

    def _risk(self, s):
        """Fuel-equivalent cost of the hazards on the remaining route."""
        risk = self.DAMAGE_RISK * s.damage
        for i, p in enumerate(s.path_ahead(80), 1):
            if p in s.fireballs:
                risk += self.FIREBALL_RISK
            t = s.pending.get(p)
            if t is not None and t <= i:      # lands before or as we arrive
                risk += self.DEBRIS_RISK
        return risk

    def _leaf(self, s):
        if s.result == ROVER or s.pos == s.m.goal:
            return 1000 + s.fuel
        d = s.route_cost()
        if s.result is not None or d == INF or d > s.fuel:
            return -1000 + (s.fuel - (d if d != INF else 100)) - self._risk(s)
        return s.fuel - d - self._risk(s)

    def _ordered(self, s):
        d = s.dist()
        return sorted(s.rover_moves(), key=lambda a: (
            s.r.enter_cost(s.blk.get(a[1], OPEN)) + d.get(a[1], INF)
            + (self.FIREBALL_RISK if a[1] in s.fireballs else 0) if a[1] else INF))

    def _alien_options(self, s):
        opts = [None]
        for p in s.path_ahead(10):
            if not s.can_target(p):
                continue
            b = s.blk.get(p, OPEN)
            opts.append(('boulder', p) if b == OPEN else ('debris', p))
            if b == OPEN:
                opts.append(('debris', p))
            if len(opts) > self.alien_branch:
                break
        return opts

    def _value(self, s, depth, alpha, beta):
        self.nodes += 1
        if s.result is not None or depth <= 0:
            return self._leaf(s)
        if s.turn == ROVER:
            val = -INF
            for a in self._ordered(s):
                c = s.clone()
                c.apply_rover(a)
                val = max(val, self._value(c, depth - 1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        val = INF
        for a in self._alien_options(s):
            c = s.clone()
            c.apply_alien(a)
            val = min(val, self._value(c, depth, alpha, beta))
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val


# --------------------------------------------------------------------- fuzzy
try:
    import skfuzzy as fuzz
except ImportError:  # small stand-in so the game still runs without scipy
    class fuzz:
        @staticmethod
        def trimf(x, abc):
            a, b, c = abc
            y = np.zeros_like(x, dtype=float)
            if b > a:
                y = np.where((x >= a) & (x <= b), (x - a) / (b - a), y)
            if c > b:
                y = np.where((x >= b) & (x <= c), (c - x) / (c - b), y)
            y[x == b] = 1.0
            return y

        @staticmethod
        def interp_membership(x, mf, v):
            return float(np.interp(v, x, mf))

        @staticmethod
        def defuzz(x, mf, mode):
            s = mf.sum()
            return float((x * mf).sum() / s) if s > 0 else 0.5


class FuzzyRover(Agent):
    """R5: Mamdani fuzzy controller sets a caution level in [0, 1] from
    fuel margin, detour length and incoming debris. A* then plans with
    extra cost on soft boulders (they may harden) and debris landing zones,
    scaled by caution: so the same planner sprints, detours or dodges."""
    name = 'Fuzzy'
    side = ROVER

    SOFT_FEAR = 6     # max extra cost on a soft boulder
    DEBRIS_FEAR = 15  # max extra cost on a debris landing tile

    def __init__(self):
        super().__init__()
        self.xm = np.linspace(0, 1, 101)      # fuel margin (slack / fuel)
        self.xd = np.linspace(0, 20, 101)     # detour length (fuel)
        self.xn = np.linspace(0, 10, 101)     # distance to incoming debris on route
        self.xc = np.linspace(0, 1, 101)      # caution
        t = fuzz.trimf
        self.margin = {'low': t(self.xm, [0, 0, 0.25]), 'mid': t(self.xm, [0.1, 0.3, 0.5]),
                       'high': t(self.xm, [0.35, 1, 1])}
        self.detour = {'short': t(self.xd, [0, 0, 6]), 'mid': t(self.xd, [3, 8, 13]),
                       'long': t(self.xd, [10, 20, 20])}
        self.near = {'near': t(self.xn, [0, 0, 3]), 'far': t(self.xn, [2, 10, 10])}
        self.caution = {'low': t(self.xc, [0, 0, 0.4]), 'mid': t(self.xc, [0.2, 0.5, 0.8]),
                        'high': t(self.xc, [0.6, 1, 1])}

    def decide_caution(self, margin, detour, near):
        mu = lambda sets, x, v: {k: fuzz.interp_membership(x, mf, v) for k, mf in sets.items()}
        m = mu(self.margin, self.xm, min(max(margin, 0), 1))
        d = mu(self.detour, self.xd, min(detour, 20))
        n = mu(self.near, self.xn, min(near, 10))
        rules = [
            (m['low'], 'low'),                           # no slack: sprint
            (min(m['high'], d['short']), 'high'),         # cheap to play safe
            (min(m['mid'], d['mid']), 'mid'),
            (min(m['mid'], d['short']), 'mid'),
            (d['long'], 'low'),                           # detour too long: shoot through
            (n['near'], 'high'),                          # debris about to land on route
            (min(m['high'], n['far'], d['mid']), 'mid'),
        ]
        agg = np.zeros_like(self.xc)
        for w, out in rules:
            agg = np.maximum(agg, np.minimum(w, self.caution[out]))
        if agg.sum() == 0:
            return 0.0
        return float(fuzz.defuzz(self.xc, agg, 'centroid'))

    def act(self, v):
        c = [0]
        cost, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, counter=c)
        softs = {p for p, b in v.belief.items() if b == SOFT}
        avoid = {p: 1000 for p in softs}
        fireballs = set(getattr(v, 'fireballs', {}))
        for p in fireballs:
            avoid[p] = max(avoid.get(p, 0), 1000)
        cost_avoid, _ = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=avoid, counter=c)
        detour = 20 if cost_avoid >= 1000 or cost == INF else cost_avoid - cost
        on_route = set(path)
        near = min((manhattan(v.rover, p) for p in list(v.pending) + list(fireballs)
                    if p in on_route or any(q in on_route for q in v.map.nbrs.get(p, ()))), default=10)
        margin = (v.fuel - cost) / max(v.fuel, 1) if cost != INF else 0
        k = self.decide_caution(margin, detour, near)
        extra = dict(self.tie_noise(v.map))
        for p in softs:
            extra[p] = extra.get(p, 0) + k * self.SOFT_FEAR
        for p, ttl in v.pending.items():
            extra[p] = extra.get(p, 0) + k * self.DEBRIS_FEAR * (2 if ttl == 1 else 1)
        for p in getattr(v, 'fireballs', {}):
            extra[p] = extra.get(p, 0) + k * self.DEBRIS_FEAR
        _, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=extra, counter=c)
        self.nodes += c[0]
        self.info = f'caution {k:.2f}'
        return path_action(v, path[1] if len(path) > 1 else None)


ROVERS = [AStarRover, MinimaxRover, FuzzyRover]

# ---------------------------------------------------------------- escape intelligence
class EscapeIntelligenceRover(Agent):
    """Hybrid rover: A* routing + BFS connectivity + Fuzzy risk + Minimax defence."""
    name = 'Escape Intelligence'
    side = ROVER

    def __init__(self, depth=3):
        super().__init__()
        self.minimax = MinimaxRover(depth=depth)
        self.fuzzy = FuzzyRover()

    def act(self, v):
        # A*: cheapest route
        cost, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules)
        # BFS: is an alternative corridor still available?
        dist = bfs_moves(v.map, v.belief, v.map.goal)
        alt_routes = 1 if v.rover in dist else 0
        # Fuzzy risk evaluation
        softs = {p for p, b in v.belief.items() if b == SOFT}
        avoid = {p: 1000 for p in softs}
        cost_avoid, _ = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=avoid)
        detour = 20 if cost_avoid >= 1000 or cost == INF else cost_avoid - cost
        on_route = set(path)
        fireballs = set(getattr(v, 'fireballs', {}))
        near = min((manhattan(v.rover, p) for p in list(v.pending) + list(fireballs)
                    if p in on_route or any(q in on_route for q in v.map.nbrs.get(p, ()))), default=10)
        margin = (v.fuel - cost) / max(v.fuel, 1) if cost != INF else 0
        k = self.fuzzy.decide_caution(margin, detour, near)
        # Minimax defensive planning with depth scaled by risk
        self.minimax.depth = 2 if k > 0.6 else 3
        a = self.minimax.act(v)
        self.nodes = self.minimax.nodes
        self.info = f'EI caution {k:.2f} alt {alt_routes}'
        return a

ROVERS.append(EscapeIntelligenceRover)
