import random
from .config import OPEN, SOFT, HARD, INF
from .graph import cost_to_go, manhattan

ROVER, ALIEN = 'rover', 'alien'


class Sim:
    __slots__ = ('m', 'r', 'pos', 'fuel', 'tick', 'budget', 'blk', 'pending',
                 'fireballs', 'damage', 'result', 'turn', 'cache', 'counter',
                 'rng', 'next_strike')

    def __init__(self, m, rules, pos, fuel, tick, budget, blk, pending, fireballs=None, damage=0, turn=ROVER,
                  rng=None, next_strike=None, cache=None, counter=None):
        self.m, self.r = m, rules
        self.pos, self.fuel, self.tick, self.budget = pos, fuel, tick, budget
        self.blk, self.pending = blk, pending
        self.fireballs = {} if fireballs is None else dict(fireballs)
        self.damage = damage
        self.result = None
        self.turn = turn
        self.rng = random.Random()
        if rng is not None:
            self.rng.setstate(rng.getstate())
        T = rules.alien_every
        self.next_strike = next_strike if next_strike is not None else (T if not rules.random_events else None)
        if self.next_strike is None:
            # mirror GameState: first strike between 1 and T
            self.next_strike = self.rng.randint(1, T) if rules.random_events else T
        self.cache = {} if cache is None else cache
        self.counter = counter if counter is not None else [0]

    @classmethod
    def from_state(cls, s, cache=None, counter=None):
        sim = cls(s.map, s.rules, s.rover, s.fuel, s.tick, s.budget, dict(s.blk),
                   dict(s.pending), dict(s.fireballs), s.damage, s.turn,
                   rng=s.rng, next_strike=s.next_strike, cache=cache, counter=counter)
        return sim

    @classmethod
    def from_view(cls, v, cache=None, counter=None):
        # rover has no rng; use a fresh deterministic rng (events fixed by rules)
        return cls(v.map, v.rules, v.rover, v.fuel, v.tick, v.budget, dict(v.belief),
                   dict(v.pending), dict(getattr(v, 'fireballs', {})), getattr(v, 'damage', 0), ROVER,
                   rng=random.Random(0), next_strike=v.tick + v.rules.alien_every, cache=cache, counter=counter)

    def clone(self):
        s = Sim(self.m, self.r, self.pos, self.fuel, self.tick, self.budget,
                dict(self.blk), dict(self.pending), dict(self.fireballs), self.damage, self.turn,
                rng=self.rng, next_strike=self.next_strike, cache=self.cache, counter=self.counter)
        s.result = self.result
        return s

    def strike_gap(self):
        T = self.r.alien_every
        j = max(1, T // 2)
        return max(1, self.rng.randint(T - j, T + j)) if self.r.random_events else T

    def dist(self):
        key = frozenset(self.blk.items())
        d = self.cache.get(key)
        if d is None:
            d = cost_to_go(self.m, self.blk, self.m.goal, self.r, counter=self.counter)
            self.cache[key] = d
        return d

    def route_cost(self):
        return self.dist().get(self.pos, INF)

    # rover ------------------------------------------------------------------
    def rover_moves(self):
        acts = []
        for v in self.m.nbrs[self.pos]:
            b = self.blk.get(v, OPEN)
            if b == OPEN:
                acts.append(('move', v))
            elif b == SOFT:
                acts.append(('shoot', v))
        acts.append(('wait', None))
        return acts

    def apply_rover(self, a):
        kind, t = a
        if kind == 'move':
            self.pos = t
            self.fuel -= self.r.move_cost
            if t in self.fireballs:
                self.fuel -= self.r.fireball_damage
                self.damage = min(self.r.max_damage, self.damage + self.r.fireball_damage)
        elif kind == 'shoot':
            del self.blk[t]
            self.fuel -= self.r.shot_cost
        else:
            self.fuel -= self.r.move_cost
        self.tick += 1
        for p in sorted(self.pending):
            self.pending[p] -= 1
            if self.pending[p] > 0:
                continue
            del self.pending[p]
            if p == self.pos:
                d = self.r.debris_damage
                dmg = self.rng.randint(d - 2, d + 2) if self.r.random_events else d
                self.fuel -= dmg
            elif self.blk.get(p, OPEN) == SOFT:
                self.blk[p] = HARD
            else:
                self.blk[p] = SOFT
        for p in list(self.fireballs):
            self.fireballs[p] -= 1
            if self.fireballs[p] <= 0:
                del self.fireballs[p]
        if not self.check_end() and self.budget > 0 and self.tick >= self.next_strike:
            self.turn = ALIEN

    # alien ------------------------------------------------------------------
    def can_target(self, p):
        return (not self.m.protected(p) and manhattan(p, self.pos) > 1
                and p not in self.pending and p not in self.fireballs and self.blk.get(p, OPEN) != HARD)

    def apply_alien(self, a):
        if a is not None:
            kind, p = a
            self.budget -= 1
            if kind == 'boulder':
                self.blk[p] = SOFT
            elif kind == 'fireball':
                self.fireballs[p] = self.r.fireball_ttl
            else:
                d = self.r.debris_delay
                self.pending[p] = self.rng.randint(max(1, d - 1), d + 1) if self.r.random_events else d
        self.next_strike = self.tick + self.strike_gap()
        self.turn = ROVER
        self.check_end()

    def check_end(self):
        if self.pos == self.m.goal:
            self.result = ROVER
        elif self.fuel <= 0:
            self.result = ALIEN
        elif self.damage >= self.r.max_damage:
            self.result = ALIEN
        else:
            c = self.route_cost()
            if c == INF or c > self.fuel:
                self.result = ALIEN
        return self.result is not None

    def greedy_rover_action(self):
        d = self.dist()
        best, best_v = ('wait', None), INF
        for v in self.m.nbrs[self.pos]:
            b = self.blk.get(v, OPEN)
            if b == HARD:
                continue
            val = self.r.enter_cost(b) + d.get(v, INF)
            if val < best_v:
                best_v = val
                best = ('shoot', v) if b == SOFT else ('move', v)
        return best

    def path_ahead(self, limit=12):
        d = self.dist()
        out, p, seen = [], self.pos, {self.pos}
        while p != self.m.goal and len(out) < limit:
            nxt = min((v for v in self.m.nbrs[p] if v not in seen),
                      key=lambda v: self.r.enter_cost(self.blk.get(v, OPEN)) + d.get(v, INF),
                      default=None)
            if nxt is None or d.get(nxt, INF) == INF:
                break
            out.append(nxt)
            seen.add(nxt)
            p = nxt
        return out