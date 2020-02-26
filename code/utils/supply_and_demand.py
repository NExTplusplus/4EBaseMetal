import os
import sys
import numpy as np
import pandas as pd

def SupplyDemandSpread(supply,demand, version = "v2"):
  if version == "v1":
    spread = supply - demand
    spread = np.true_divide(spread[1:],spread[:-1])
  elif version == "v2":
    spread = np.log(supply) - np.log(demand)
  return spread