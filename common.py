import numpy as np
import pandas as pd

NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILD_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
FEATURES = NUM_COLS + WILD_COLS + SOIL_COLS
TARGET = "Cover_Type"

COVER_NAMES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
}
WILD_NAMES = {
    1: "Rawah", 2: "Neota", 3: "Comanche Peak", 4: "Cache la Poudre",
}


def load_covtype(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, header=None)
    df.columns = FEATURES + [TARGET]
    return df.astype("int16")   # ~4x less RAM than int64


def build_row(num_values: dict, wilderness: int, soil: int) -> pd.DataFrame:
    """Build a single-row DataFrame with the exact 54 training columns."""
    row = {c: 0 for c in FEATURES}
    row.update(num_values)
    row[f"Wilderness_Area{wilderness}"] = 1
    row[f"Soil_Type{soil}"] = 1
    return pd.DataFrame([row], columns=FEATURES)
