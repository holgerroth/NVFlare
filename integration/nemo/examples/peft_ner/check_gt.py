#!/usr/bin/python3

import os
import json
import numpy as np

datafiles = ["/media/hroth/workspace/Data/NLP/NCBI-disease/NCBI-disease-20230831T023848Z-001/NCBI-disease/4_split/site-1_val.jsonl",
             "/media/hroth/workspace/Data/NLP/NCBI-disease/NCBI-disease-20230831T023848Z-001/NCBI-disease/4_split/site-2_val.jsonl",
             "/media/hroth/workspace/Data/NLP/NCBI-disease/NCBI-disease-20230831T023848Z-001/NCBI-disease/4_split/site-3_val.jsonl",
             "/media/hroth/workspace/Data/NLP/NCBI-disease/NCBI-disease-20230831T023848Z-001/NCBI-disease/4_split/site-4_val.jsonl"]

gt = []
lines_lenghts = []
for df in datafiles:
    with open(df, 'r') as f:
        lines = f.readlines()
    lines_lenghts.append(len(lines))

    for l in lines:
        data = json.loads(l)
        for d in data['output']:
            gt.append(d)

print(f"Loaded {len(gt)} gt values from {lines_lenghts} lines.")
print("Unique gt values", np.unique(gt))
