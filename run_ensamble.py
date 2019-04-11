import numpy as np
import pandas as pd
import datetime
import logging
import argparse
import time
import sys
import os

import warnings
warnings.filterwarnings('ignore')

now = datetime.datetime.now()
filename = 'log_ensamble_{0:%Y%m%d%H%M%S}.log'.format(now)
logging.basicConfig(
    filename='./logs/' + filename, level=logging.DEBUG
)
logging.debug('./logs/' + filename)


outputs = os.listdir("./data/output/")

sub = pd.read_csv("./data/input/sample_submission.csv")
pred = np.zeros(len(sub))
files = []

for output in outputs:
    if "sub_" in output and "ensamble" not in output:
        if (output.split("_")[1], output.split("_")[1]) not in files:
            logging.debug(output)
            pred = pd.read_csv("./data/output/"+output, usecols=['target'])['target'].values / len(outputs)
            files.append((output.split("_")[1], output.split("_")[1]))

sub['target'] = pred

sub.to_csv(
    './data/output/sub_ensamble_{0:%Y%m%d%H%M%S}.csv'.format(now),
    index=False
)
