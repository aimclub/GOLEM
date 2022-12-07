import datetime
import time

from golem.core.optimisers.timer import OptimisationTimer


def test_composition_timer():
    generation_num = 100
    reached = False
    start = datetime.datetime.now()
    with OptimisationTimer(timeout=datetime.timedelta(minutes=0.01)) as timer:
        for generation in range(generation_num):
            time.sleep(1)
            if timer.is_time_limit_reached(iteration_num=generation):
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1
