# Autonomous Escape

Autonomous Escape is a turn-based rover-versus-alien game set on Phobos. The
rover must reach the launch pad across a procedurally generated debris field;
the alien tries to make that impossible by blocking routes, creating hazards,
or draining the rover's fuel. The project includes the rules engine, map and
terrain generation, search agents, a Pygame front end, and a benchmark runner.

This document describes the implementation as it exists in the source. Some
older comments and checked-in benchmark results refer to agents and rules that
are no longer registered; those are called out below.

## Run the game

Install dependencies and start the UI:

```sh
pip install -r requirements.txt
python main.py --mode ai --rover 'A*' --alien MCTS
```

To start directly in a human-controlled mode:

```sh
python main.py --mode rover --rover 'A*' --alien MCTS --play
python main.py --mode alien --rover 'A*' --alien MCTS --play
```

`--mode` accepts `ai`, `rover`, or `alien`; `--rover` and `--alien` select
registered agents; `--play` skips the menu; `--speed` sets the AI playback
speed. The `--seed` option selects a region's visual theme, not the random
map/game seed.

The CLI and `App` constructor default to rover `A*` and alien `Greedy`.

## What is in the codebase

```text
main.py                    UI launcher
bench.py                   benchmark command-line interface
escape/config.py           rule values and random rule selection
escape/mapgen.py           logical map construction and validation
escape/graph.py            A*, BFS, reverse Dijkstra, vertex cuts
escape/state.py            authoritative game state and turn rules
escape/fastsim.py          compact simulator for search agents
escape/agents/base.py      agent interface and route-to-action helper
escape/agents/rovers.py    rover decision policies
escape/agents/aliens.py    alien decision policies
escape/bench.py            game runner, statistics, heatmap
ui/app.py                  menus, input, turn loop, rendering orchestration
ui/terrain.py              procedural visual ground
ui/sprites.py              rover, rock, saucer, and base sprites
results/                   checked-in benchmark outputs (legacy; see below)
tests/                     rules, graph, agent, and UI tests
```

## A game from start to finish

The normal UI path is:

1. `App.start_game()` calls `GameState.random_game()` with no explicit seed.
2. `GameState.random_game()` picks a fresh 32-bit game seed, calls
   `mapgen.generate_random(seed)`, then draws rules with `random_rules(seed,
   map)`.
3. `GameState` initializes the rover at the map's start, computes the
   shortest initial route, derives starting fuel from that route, initializes
   the alien budget and strike schedule, then records what the rover can see.
4. On each rover decision, an agent receives `rover_view()`; on each alien
   decision, an agent receives the full `GameState`. The selected action is
   passed to the rules engine, which updates the state and checks for an end
   condition.
5. The UI animates the resulting move/effect and repeats until the rover wins
   or the alien wins.

The rules engine—not the UI or an agent—owns the authoritative game state. The
main API entry points are `GameState.random_game(seed)` for the UI-style random
game, `GameState.new(map_seed, rules, game_seed)` for the fixed-configuration
map generator, `state.rover_view()` for a rover observation, and
`step(state, rover_agent, alien_agent)` for a single AI decision.

## Maps: logical layout versus appearance

### Random maps used by the game UI

`mapgen.generate_random(game_seed)` builds the logical grid deterministically
from `game_seed`:

- It chooses an odd size from 41 through 61 and places the base and pad on
  opposite sides or opposite corners.
- `_field()` creates a noise-based rock field. The selected fill fraction is
  0.42–0.55, with one or two smoothing passes to form clusters.
- It adds one or two features selected from crater rings, bars, ridges, rock
  islands, or scattered rocks. (The generator's feature selection does not
  currently include plazas.)
- `_carve()` lays 5–9 winding channels through the rock field, with zero to two
  random waypoints per channel. It biases new channels away from previously
  carved ones, but permits some overlap.
- It adds an outer wall, clears circular landing areas around the endpoints,
  and fills disconnected open pockets back in as walls.
- It tries up to 150 candidates, seeking a route at least 1.1 times the
  Manhattan distance and at least three vertex-disjoint routes from base to
  pad zone. If it cannot find a qualifying candidate, it returns its best
  candidate with at least two routes when available; otherwise it falls back
  to the regular generator.

The source also contains a recursive-backtracker `_maze()` helper, but the
current `generate_random()` path does not call it. The random maps in play are
therefore noise-grown rock fields with carved channels and overlays, not
recursive-backtracker mazes.

### Regular maps used by tests and benchmarks

`GameState.new(map_seed, rules, game_seed)` calls `mapgen.generate()` instead.
With the default `Rules`, these maps use a fixed 51×51 grid, fixed endpoint
positions, ridges, carved channels, and route/length acceptance checks. This
is a different generator from the random-size map used by the UI. The
benchmark section explains why that distinction matters.

### Region themes and terrain rendering

The five region names—Stickney Rim, Kepler Grooves, Limtoc Basin, Reldresal
Maze, and Drunlo Serpentine—select one of five terrain themes (iron, basalt,
pale, boulder, or ice). A theme changes colours, lighting, and decorative
terrain; it does not force the generated logical map to use the similarly
named feature.

`ui/terrain.py` turns the logical grid into a lit pixel surface using seeded
noise, height fields, craters, rock stamps, hill shading, and cast shadows.
The logical walls still determine collision and pathfinding. Decorative
pebbles on open tiles are not obstacles, and visual crater/rock detail does
not change movement rules. `ui/sprites.py` draws the rover, boulders, saucer,
and base in code with supersampling; no sprite image files are required for
those objects.

The UI displays the actual `game_seed` as `#...`. Recreate the generated map,
rules, and random events with `GameState.random_game(game_seed)` in Python. The
in-game `R` key currently starts a fresh random game rather than replaying that
seed, despite the UI's “replay” label. `N` also starts a fresh game and moves
to the next region theme.

## Rules and turn order

The defaults live in `escape/config.py`; each random UI game replaces several
of them with values drawn from its game seed.

### Starting conditions

- The rover starts on the base tile; no boulders are initially placed.
- Starting fuel is `round(fuel_mult * shortest_route_cost)`. Random games draw
  `fuel_mult` from 1.6–2.2. With an empty map, route cost is the number of
  orthogonal steps.
- The alien's random-game budget is based on the map's vertex cut (capped at
  eight for this calculation), with small random adjustments. The cut
  approximates how many tiles must be blocked to separate the start from the
  pad zone.
- The nominal alien interval is drawn from 3–7 rover turns. The first strike
  is scheduled randomly within that interval; subsequent gaps vary by up to
  half the nominal interval.

### Rover decision and environment update

Each rover action is one tick:

1. Move to an adjacent open tile for `move_cost` fuel (default 1), wait for the
   same cost, or shoot an adjacent soft boulder for `shot_cost` fuel (default
   2). Shooting destroys the boulder and leaves the rover in place.
2. If a move enters a tile with an active fireball, add fireball damage
   (default 10). Fireballs lose one unit of lifetime after each rover action.
3. Decrement all pending debris counters. On landing, debris on the rover
   costs fuel (randomly 4–8 in a random game); debris on a soft boulder makes
   it hard; debris on open ground creates a soft boulder.
4. Refresh rover sensing and check whether the game has ended. If the alien's
   next strike is due and its budget is not exhausted, hand over the decision.

A soft boulder can be shot; a hard boulder is a wall for movement and cannot
be shot. A strike on open ground can drop a boulder immediately, place a
fireball hazard, or launch delayed debris. In random games, debris delay is
1–3 rover actions. The alien can hold a strike; holding does not spend budget,
but the next strike is still scheduled.

Alien targets must be open map tiles outside the protected zones around the
base and pad, more than one Manhattan step from the rover, and not already
occupied by pending debris or a fireball. Hard boulders cannot be targeted.
Boulders and fireballs can only be placed on open ground; debris may also
harden a soft boulder.

### End conditions and scoring

After actions, the engine checks in this order: the rover reached the pad;
fuel is at or below zero; accumulated fireball damage reached the damage
limit (default 100); no route exists; or the cheapest remaining route costs
more fuel than the rover has. The last two conditions give the alien a win
without requiring every open tile to be blocked. A rover win scores
`100 * remaining_fuel / starting_fuel` (clamped at zero); a loss scores zero.

Route affordability uses the current boulder layout and movement/shot costs.
It is not a forecast of future alien strikes or a full simulation of hazard
risk.

### What each side knows

The rover's observation is a deliberately restricted view:

- The complete wall map is known from orbit.
- The rover has a circular radar with radius five tiles. Boulders in radar are
  added to its belief; previously seen boulders remain remembered after moving
  away, unless later observed gone.
- Pending debris is visible anywhere, including its target and countdown.
- The rover observation also carries fuel, damage, tick, strike budget, and
  shot counters. The alien receives the complete state, including all
  boulders and pending hazards.

`GameState.rover_view()` includes fireball state, but the current UI does not
render fireballs. Rover-mode fog is always drawn; the `F` toggle only removes
fog for the omniscient AI/alien view.

## How decisions are made

All agents implement `act(observation)` and return an action tuple such as
`('move', tile)`, `('shoot', tile)`, `('boulder', tile)`, `('fireball', tile)`,
`('debris', tile)`, or `None`. Rover agents get only the observation above; alien agents see the
full state. `Agent.tie_noise()` gives the A* and Fuzzy rovers a tiny seeded per-tile cost
so equally good routes can vary. Since these small penalties accumulate
along a route, they can also slightly perturb route preferences.

### Shared path and graph routines

`escape/graph.py` supplies the core calculations:

- **A-star** finds the least fuel-cost route. Entering an open tile costs
  one move; entering a soft-boulder tile costs a shot plus a move; a hard tile
  is impassable. The heuristic is Manhattan distance.
- **Reverse Dijkstra** (`cost_to_go`) computes the cheapest cost from every
  tile to a destination. The search simulator caches these maps by boulder
  layout so repeated route-cost checks are inexpensive.
- **BFS** computes move distance through passable tiles.
- **Minimum vertex cut** splits each grid tile into an in-node and out-node.
  Cuttable tiles have capacity one and protected/uncuttable tiles and links
  have a very large capacity. Edmonds–Karp max-flow then finds a small set of
  alien-placeable tiles separating the rover from the pad zone. `disjoint_routes`
  reuses this construction to estimate route redundancy.

### Rover agents

| Registered name | Decision process |
|---|---|
| `A*` | Runs weighted A* on the rover's belief map every turn and follows the first step. If that step is a soft boulder, the action is to shoot it first. |
| `Minimax` | Searches rover actions and likely alien replies with alpha-beta pruning (default depth 4, alien branch target 6). The alien model considers holding or attacking among the first ten predicted route tiles. A leaf is valued by remaining fuel minus cheapest route cost; a reached pad is strongly rewarded and a loss strongly penalized. |
| `Fuzzy` | Computes fuel margin, detour cost when avoiding known soft boulders, and proximity of incoming hazards to its route. Mamdani rules generally favor caution when fuel slack is ample and detours are short, but favor pressing on when slack is low or a detour is long; nearby debris raises caution. A* then adds caution-scaled costs for soft boulders, imminent debris, and known fireballs. |
| `Escape Intelligence` | Computes A* route cost, a BFS reachability flag, Fuzzy caution, and a Minimax action. The BFS flag is binary (not a route count) and is only reported in the agent's info string; Fuzzy caution controls whether the embedded Minimax search uses depth 2 or 3. |

### Alien agents

| Registered name | Decision process |
|---|---|
| `Random` | Picks a legal target tile; it drops a boulder on open ground and launches debris at a soft boulder. |
| `Greedy` | Looks at the first three legal tiles on the rover's cheapest route, chooses among them with weights 3:2:1, then uses a fireball about half the time on open ground; otherwise it blocks/hardens with boulders or debris. |
| `MCTS` | Monte Carlo tree search over alien actions using UCT. Candidate targets come from the predicted route and existing soft boulders. During search it advances a greedy cost-to-go rover; rollouts use a cheaper route-blocking policy. Terminal wins/losses score 1/0, and nonterminal states are scored from the fuel-versus-route-cost margin. Class defaults are 150 iterations, a 0.6-second agent limit, and a four-alien-turn horizon. The UI uses a 0.5-second limit; the benchmark CLI uses 10 seconds and exposes the iteration count as `--mcts-iters`. |
| `Strategic Hunter` | Subclasses MCTS and replaces candidate generation with a broader set of fireball, boulder, and debris actions on the predicted route and soft boulders. Despite the class docstring's hybrid description, its current implementation does not add separate explicit minimax, max-flow, or CSP decision stages. |

### Search model limitation

`escape/fastsim.py` is used by Minimax and MCTS to simulate future decisions.
It agrees with `GameState` under fixed events, as tested with
`random_events=False`, but it does not model the default random event schedule:
it triggers alien turns on `tick % alien_every`, uses fixed debris delay and
fixed debris-hit damage. The real engine uses seeded variable strike gaps,
debris delays, and damage. Consequently, search agents reason over an
approximation of random games rather than an exact replay of their timing and
damage distributions.

## UI modes and controls

In the menu, Up/Down or W/S selects a row; Left/Right or A/D cycles the mode,
agent, region, or speed; Enter/Space launches. `R` selects a random region;
`Esc` quits from the menu.

| Mode | Player action |
|---|---|
| AI vs AI | Watch the selected rover and alien agents. Space pauses; `→` or `.` advances one move; `+` / `-` changes playback speed. |
| You are the Rover | Arrows / WASD move. Bumping a loose boulder shoots it instead. Space waits. |
| You are the Alien | Left-click drops a boulder on open ground or launches debris at a boulder. Right-click launches debris at the selected tile. Space holds the strike. |

The alien-player UI does not offer fireballs, though AI agents can use them.
Camera and overlays: mouse wheel zooms at the cursor; middle-drag pans; right-
drag also pans except in Alien mode, where right-click is an action. `Z` fits
the map, `V` toggles rover-follow, and `[` / `]` zoom. `F` toggles fog where
allowed, `P` toggles the planned rover route, `C` toggles the cut display, and
`G` toggles the grid. `Esc` returns to the menu.

The cut-display toggle currently has no computed cut to draw: the UI overlay
path initializes the cut as empty. Likewise, the UI labels `R` and the end
screen as “replay,” but the key actually launches a fresh random game. `N`
starts a fresh game and advances the region theme.

The UI is a 60-FPS presentation loop. It advances the engine on a timer, then
interpolates rover motion and draws tracks, impact/laser effects, incoming
debris reticles, route lines, and HUD information. These visual effects do not
alter the rules state.

## Benchmarks and results

Run all registered agent pairings over map seeds 100–299 by default:

```sh
python bench.py
python bench.py --seeds 100-139 --aliens MCTS,Greedy
python bench.py --rovers "Escape Intelligence" --aliens "Strategic Hunter" --jobs 1
```

Useful options include `--rovers`, `--aliens`, `--jobs`, `--mcts-iters`, and
`--out`. For each map seed, the benchmark calls `GameState.new()` with the
default fixed-size map rules; it does **not** use the UI's 41–61 random-map
generator. Each game receives its own random `game_seed` for events and agent
choices. Thus pairings share the same logical map seed, but are not guaranteed
to share the same random event sequence.

The runner records the map seed and game seed, winner/reason, score, turns,
shots, debris hits, alien actions, average decision time, and average search
nodes. The summary reports alien win rates with Wilson 95% intervals, sealing
rates, escape scores, and performance metrics. It writes:

- `results/games.csv` — individual games
- `results/summary.csv` — aggregate pairings
- `results/alien_win_heatmap.png` — alien win-rate visualization

The checked-in CSVs are legacy artifacts: their roster includes old names such
as Flood Fill, D* Lite, and Core, and the individual-game CSV predates the
`game_seed` field. Rerun the benchmark to produce results for the current
registry and replayable event seeds.

## Tests

Run the test suite with:

```sh
python -m pytest
```

The tests cover deterministic map generation, fuel and action costs, alien
legality, debris timing and effects, end conditions, radar/belief behavior,
random-game replay from a seed, fixed-event `Sim` equivalence, path-cost
cross-checks, max-flow cuts compared with NetworkX, all registered rover/alien
pairings finishing, and headless UI interactions. The simulator-equivalence
test explicitly disables random events, so it does not verify the random-event
approximation described above.
