import pandas as pd
import json
import os

def save_twiss(tw, basename="twiss"):
    """
    Save TwissTable (per-element optics + all scalar attributes).
    
    Parameters
    ----------
    tw : xtrack.TwissTable
        Result of line.twiss()
    basename : str
        Base filename (without extension).
    """
    # Save per-element optics
    tw.to_pandas().to_csv(f"{basename}.csv", index=False)

    # Collect scalars (things that are not array-like)
    scalars = {}
    for k, v in tw._data.items():
        # Keep numbers, strings, bools
        if not hasattr(v, "__len__") or isinstance(v, str):
            try:
                json.dumps(v)  # check if serializable
                scalars[k] = v
            except TypeError:
                pass  # skip un-serializable objects

    # Save scalars
    with open(f"{basename}_scalars.json", "w") as f:
        json.dump(scalars, f, indent=2)


def load_twiss(basename="twiss"):
    """
    Load previously saved Twiss optics (CSV + scalars JSON).
    
    Returns
    -------
    df : pandas.DataFrame
        Per-element optics
    scalars : dict
        Dictionary of scalar values
    """
    df = pd.read_csv(f"{basename}.csv")
    with open(f"{basename}_scalars.json") as f:
        scalars = json.load(f)
    return df, scalars