I like type annotations, but sometimes they can be very ugly.

import pandas as pddef f(inp):
 df = pd.read_csv(inp)
 return df

to

import pandas as pd
from pathlib import Path
from typing import AnyStr, IO, Uniondef f(inp: Union[str, Path, IO[AnyStr]]) -> pd.Dataframe:
 df = pd.read_csv(inp)
 return df

https://github.com/pandas-dev/pandas/blob/1ce1c3c1ef9894bf1ba79805f37514291f52a9da/pandas/_typing.py#L48:1
