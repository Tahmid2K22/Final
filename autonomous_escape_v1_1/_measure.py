import random, time, sys
from escape.state import GameState, ROVER, ALIEN
from escape.agents import make_rover, make_alien


def play(game_seed, rover_name, alien_name, budget=False):
    s = GameState.random_game(game_seed)
    if budget:
        s.budget = s.budget0 = s.rules.alien_budget
    rover = make_rover(rover_name, game_seed)
    alien = make_alien(alien_name, game_seed)
    rover.reset(); alien.reset()
    t0 = time.perf_counter()
    while s.result is None:
        if s.turn == ALIEN:
            a = alien.act(s)
            s.apply_alien(a)
        else:
            a = rover.act(s.rover_view())
            s.apply_rover(a)
    return s, time.perf_counter() - t0


if __name__ == '__main__':
    rn, an, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    use_budget = len(sys.argv) > 4 and sys.argv[4] == 'budget'
    wins = {ROVER: 0, ALIEN: 0}
    reasons = {}
    t0 = time.perf_counter()
    for i in range(n):
        seed = 1200 + i
        s, dt = play(seed, rn, an, use_budget)
        wins[s.result] += 1
        reasons[s.reason] = reasons.get(s.reason, 0) + 1
        print(f'{seed}: {s.result:5s} {s.reason:32s} tick={s.tick:4d} fuel={s.fuel:5d} '
              f'budget={s.budget0 - s.budget} {dt:5.1f}s', flush=True)
    print(f'{rn} vs {an}: rover {wins[ROVER]}/{n}  alien {wins[ALIEN]}/{n}')
    print(reasons)
    print(f'total {time.perf_counter() - t0:.1f}s')
