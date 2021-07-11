from functools import wraps


class Strategy:
    @classmethod
    def wrap(cls, func):
        return lambda *a, **k: Strategy(wraps(func)(lambda: func(*a, **k)))

    def __init__(self, strategy):
        self.strategy = strategy

    def run(self, return_condition=False):
        gen = self.strategy()
        if not next(gen):
            if return_condition:
                return False
            return None
        try:
            next(gen)
            assert 0
        except StopIteration as e:
            if return_condition:
                return True
            return e.value

    def check_condition(self):
        gen = self.strategy()
        return next(gen)

    def condition(self, condition):
        def f(self=self, condition=condition):
            if not condition():
                yield False
                assert 0
            it = self.strategy()
            yield next(it)
            try:
                next(it)
                assert 0
            except StopIteration as e:
                return e.value

        return Strategy(f)

    def until(self, agent, condition):
        def f():
            if not condition():
                yield False
                assert 0
            yield True

        return self.preempt(agent, [Strategy(f)], continue_after_preemption=False)

    def before(self, strategy):
        def f(self=self, strategy=strategy):
            yielded = False
            r1, r2 = None, None

            v1 = self.strategy()
            if next(v1):
                if not yielded:
                    yielded = True
                    yield True
                try:
                    next(v1)
                    assert 0, v1
                except StopIteration as e:
                    r1 = e.value

            v2 = strategy.strategy()
            if next(v2):
                if not yielded:
                    yielded = True
                    yield True
                try:
                    next(v2)
                    assert 0, v2
                except StopIteration as e:
                    r2 = e.value

            if not yielded:
                yield False

            return (r1, r2)

        return Strategy(f)

    def preempt(self, agent, strategies, continue_after_preemption=True):
        def f(self=self, agent=agent, strategies=strategies):
            gen = self.strategy()
            condition_passed = False
            with agent.disallow_step_calling():
                condition_passed = next(gen)

            if not condition_passed:
                yield False
            yield True

            assert not agent._no_step_calls

            return agent.preempt(strategies, self, continue_after_preemption=continue_after_preemption)

        return Strategy(f)

    def repeat(self):
        def f(self=self):
            yielded = False
            val = None
            while 1:
                gen = self.strategy()
                if not next(gen):
                    if not yielded:
                        yield False
                    return

                if not yielded:
                    yielded = True
                    yield True

                try:
                    next(gen)
                    assert 0, gen
                except StopIteration as e:
                    val = e.value
            return val

        return Strategy(f)
