from data.load_rnn import load_pure_log_reg

import numpy as np
import pandas as pd
import json


load_pure_log_reg(json.load({"../../../../LME/NExT/LME/LMECopper3M.csv": ["Volume","Close.Price"],"../../../../LME/NExT/LME/LMCADY.csv": ["LMCADY"]})
                    ,"LMCADY","log_nd_return", ['2015-01-01','2015-02-01','2015-03-01'], 5, S=1)



