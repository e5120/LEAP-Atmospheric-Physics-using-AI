from leap.utils.utils import (
    setup,
    get_num_training_steps,
    build_callbacks,
    normalize,
)


IN_SCALAR_COLUMNS = [
    "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX", "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS",
    "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR", "cam_in_LWUP", "cam_in_ICEFRAC",
    "cam_in_LANDFRAC", "cam_in_OCNFRAC", "cam_in_SNOWHLAND",
]
IN_VECTOR_COLUMNS = [
    "state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O",
]
IN_AUX_COLUMNS = [
    "location", "timestamp",
    # "time",
    # "sin_time", "cos_time",
]
IN_COLUMNS = IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS
OUT_SCALAR_COLUMNS = [
    "cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC",
    "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD",
]
OUT_VECTOR_COLUMNS = [
    "ptend_t", "ptend_q0001", "ptend_q0002", "ptend_q0003",
    "ptend_u", "ptend_v",
]
OUT_COLUMNS = OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS

NUM_GRID = 384
DOWN_SAMPLING = 7
DATA_INTERVAL = 20  # [/minute]
HOUR = 60 / DATA_INTERVAL  # [point/hour]
DAY = 24 * HOUR # [point/day]
YEAR = 365 * DAY  # [point/year]

NUM_TRAIN = 10_091_520
NUM_TEST = 625_000
# NUM_OLD_TEST = 625_000
NUM_ALL = NUM_TRAIN + NUM_TEST


def get_label_columns(cols):
    label_cols = []
    for col in cols:
        if col in OUT_SCALAR_COLUMNS:
            label_cols.append(col)
        else:
            for i in range(60):
                label_cols.append(f"{col}_{i}")
    return label_cols
