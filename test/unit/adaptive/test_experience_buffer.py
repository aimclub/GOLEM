from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.graph import OptGraph, OptNode


def _make_buffer(size: int) -> ExperienceBuffer:
    inds = [Individual(OptGraph(OptNode('a'))) for _ in range(size)]
    actions = ['mutation'] * size
    rewards = [0.1] * size
    return ExperienceBuffer(inds=inds, actions=actions, rewards=rewards)


def test_split_respects_ratio_on_tiny_buffers():
    """int(len * ratio) == 0 must give an empty train part, not an inverted split."""
    for size, ratio in ((1, 0.8), (4, 0.2)):
        buffer = _make_buffer(size)
        num_train_expected = int(size * ratio)
        train, val = buffer.split(ratio=ratio)
        assert len(train) == num_train_expected
        assert len(val) == size - num_train_expected


def test_split_preserves_all_items():
    buffer = _make_buffer(10)
    train, val = buffer.split(ratio=0.8, shuffle=True)
    assert len(train) == 8
    assert len(val) == 2
