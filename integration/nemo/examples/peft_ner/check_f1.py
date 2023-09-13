#!/usr/bin/python3
import numpy as np

from nemo.collections.common.metrics.classification_accuracy import MyF1Score
from seqeval.metrics import classification_report

y_true = [['O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I'], ['O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'B', 'I'], ['O', 'O', 'O', 'O', 'O', 'O',  'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'B', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'B', 'I'], ['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O', 'I', 'I', 'O']]
y_pred = [['O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I'], ['O', 'O', 'O'], ['O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'B', 'I'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O',  'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'B', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'B', 'I'], ['B', 'I', 'O'], ['I', 'I', 'O'], ['O', 'B', 'I'], ['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O', 'B', 'O', 'O']]


print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0))
metric_dict = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0)
f1_seqeval = metric_dict["macro avg"]["f1-score"]

metric = MyF1Score()

support = 0
for pred, target in zip(y_true, y_pred):
    target = np.asarray(target)
    support += np.sum(target == "B")
    metric.update(pred=" ".join(pred), target=" ".join(target))

f1 = metric.compute()
print(f"SeqEval {f1_seqeval}, MyF1Score {f1.item()}, Diff {f1_seqeval-f1.item()}, Support {support}, N {len(y_true)}")
