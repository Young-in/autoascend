from enum import IntEnum, auto

import numpy as np

import utils
from glyph import Hunger, G
from level import Level
from strategy import Strategy
import soko_solver


class Milestone(IntEnum):
    FIND_GNOMISH_MINES = auto()
    FIND_SOKOBAN = auto()
    SOLVE_SOKOBAN = auto()
    GO_DOWN = auto() # TODO


class GlobalLogic:
    def __init__(self, agent):
        self.agent = agent
        self.milestone = Milestone(1)
        self.step_completion_log = {}  # Milestone -> (step, turn)

    @utils.debug_log('solving sokoban')
    @Strategy.wrap
    def solve_sokoban_strategy(self):
        # TODO: refactor
        if not utils.isin(self.agent.last_observation['chars'], [ord('^')]).any():
            yield False
        yield True

        wall_map = utils.isin(self.agent.current_level().objects, G.WALL)
        for smap, answer in soko_solver.maps.items():
            sokomap = soko_solver.convert_map(smap)
            offset = np.array(min(zip(*wall_map.nonzero()))) - \
                     np.array(min(zip(*(sokomap.sokomap == soko_solver.WALL).nonzero())))
            mask = wall_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                            offset[1] : offset[1] + sokomap.sokomap.shape[1]]
            if (mask & (sokomap.sokomap == soko_solver.WALL) == mask).all():
                break
        else:
            assert 0, 'sokomap not found'

        for (y, x), (dy, dx) in answer:
            boulder_map = utils.isin(self.agent.glyphs, G.BOULDER)
            mask = boulder_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                               offset[1] : offset[1] + sokomap.sokomap.shape[1]]
            ty, tx = offset[0] + y - dy, offset[1] + x - dx,
            soko_boulder_mask = sokomap.sokomap == soko_solver.BOULDER
            if self.agent.bfs()[ty, tx] != -1 and \
                    ((soko_boulder_mask | mask) == soko_boulder_mask).all() and \
                    self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER:
                self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                self.agent.move(ty + dy, tx + dx)

            sokomap.move(y, x, dy, dx)

        assert 0, 'sakomap unsolvable'

    @Strategy.wrap
    def current_strategy(self):
        if self.milestone == Milestone.FIND_GNOMISH_MINES and \
                self.agent.current_level().dungeon_number == Level.GNOMISH_MINES:
            self.milestone = Milestone(int(self.milestone) + 1)
        elif self.milestone == Milestone.FIND_SOKOBAN and \
                self.agent.current_level().dungeon_number == Level.SOKOBAN:
            self.milestone = Milestone(int(self.milestone) + 1)
        elif self.milestone == Milestone.SOLVE_SOKOBAN and \
                self.agent.current_level().key() == (Level.SOKOBAN, 1):
            self.milestone = Milestone(int(self.milestone) + 1)

        yield from \
            (self.agent.exploration.go_to_level_strategy(
                Level.GNOMISH_MINES if self.milestone == Milestone.FIND_GNOMISH_MINES else Level.SOKOBAN, 1,
                lambda y, x: (self.agent.exploration.explore1(None)
                              .preempt(self.agent, [self.agent.exploration.go_to_strategy(y, x)])
                              .until(self.agent, lambda: (self.agent.blstats.y, self.agent.blstats.x) == (y, x))
                              ),
                self.agent.exploration.explore1(None)) \
            .before(utils.assert_strategy('end'))).strategy()

    def global_strategy(self):
        return (
            self.current_strategy()
            .preempt(self.agent, [
                self.agent.exploration.explore1(0),
                self.agent.exploration.explore1(None)
                    .until(self.agent, lambda: self.agent.blstats.score >= 950 and
                                               self.agent.blstats.hitpoints >= 0.9 * self.agent.blstats.max_hitpoints)
            ])
            .preempt(self.agent, [self.solve_sokoban_strategy().condition(lambda: self.agent.current_level().dungeon_number == Level.SOKOBAN)])
            .preempt(self.agent, [
                self.agent.eat1().condition(lambda: self.agent.blstats.time % 3 == 0 and
                                                    self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY)
                                 .condition(lambda: self.milestone != Milestone.SOLVE_SOKOBAN),
                self.agent.eat_from_inventory().condition(lambda: self.milestone != Milestone.SOLVE_SOKOBAN),
            ]).preempt(self.agent, [
                self.agent.fight1(),
            ]).preempt(self.agent, [
                self.agent.emergency_strategy(),
            ])
        )