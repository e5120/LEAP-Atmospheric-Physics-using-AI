from leap.utils.utils import (
    setup,
    get_num_training_steps,
    build_callbacks,
    normalize,
)

NUM_GRID = 384
DOWN_SAMPLING = 7
DATA_INTERVAL = 20  # [/minute]
HOUR = 60 / DATA_INTERVAL  # [point/hour]
DAY = 24 * HOUR # [point/day]
YEAR = 365 * DAY  # [point/year]

NUM_TRAIN = 10_091_520
NUM_TEST = 625_000
NUM_ALL = NUM_TRAIN + NUM_TEST
