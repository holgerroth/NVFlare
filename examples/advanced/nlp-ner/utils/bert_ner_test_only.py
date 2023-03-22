# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re

import pandas as pd
import torch
from bert import BertModel
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Perform model testing by loading the best global model")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--model_path", type=str, help="Path to workspace server folder")
    parser.add_argument("--num_labels", type=int, help="Number of labels for the candidate dataset")
    return parser


def align_label(
    tokenized_inputs,
    origional_text,
    labels,
    labels_to_ids,
    label_all_tokens=False,
    tokenizer=None,
):
    null_label_id = -100
    label_ids = []
    origional_text = origional_text.split(" ")

    orig_labels_i = 0
    partially_mathced = False
    sub_str = str()
    for token_id in tokenized_inputs["input_ids"][0]:
        token_id = token_id.numpy().item()
        cur_str = tokenizer.convert_ids_to_tokens(token_id).lower()
        if (
            (token_id == tokenizer.pad_token_id)
            or (token_id == tokenizer.cls_token_id)
            or (token_id == tokenizer.sep_token_id)
        ):

            label_ids.append(null_label_id)

        elif (
            (not partially_mathced)
            and origional_text[orig_labels_i].lower().startswith(cur_str)
            and origional_text[orig_labels_i].lower() != cur_str
        ):

            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = True
            sub_str += cur_str

        elif (not partially_mathced) and origional_text[orig_labels_i].lower() == cur_str:
            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = False

        else:
            label_ids.append(null_label_id)
            sub_str += re.sub("#+", "", cur_str)
            if sub_str == origional_text[orig_labels_i - 1].lower():
                partially_mathced = False
                sub_str = ""

    return label_ids


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df, labels_to_ids, tokenizer, max_length=150):
        lb = [i.split(" ") for i in df["labels"].values.tolist()]
        txt = df["text"].values.tolist()
        self.texts = [
            tokenizer.encode_plus(
                str(i),
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            for i in txt
        ]
        self.labels = [
            align_label(t, tt, l, labels_to_ids=labels_to_ids, tokenizer=tokenizer)
            for t, tt, l in zip(self.texts, txt, lb)
        ]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


if __name__ == "__main__":
    parser = data_split_args_parser()
    args = parser.parse_args()
    device = torch.device("cuda")

    model_path = args.model_path
    data_path = args.data_path
    num_labels = args.num_labels

    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))
    # label and id conversion
    labels = []
    for x in df_test["labels"].values:
        labels.extend(x.split(" "))
    unique_labels = set(labels)
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    # model
    model = BertModel(num_labels=num_labels).to(device)
    model_weights = torch.load(os.path.join(model_path, "best_FL_global_model.pt"))
    model.load_state_dict(state_dict=model_weights["model"])
    tokenizer = model.tokenizer

    # data
    test_dataset = DataSequence(df_test, labels_to_ids, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=64, shuffle=False)

    # validate
    model.eval()
    with torch.no_grad():
        total_acc_test, total_loss_test, test_total = 0, 0, 0
        test_y_pred, test_y_true = [], []
        for test_data, test_label in test_loader:
            test_label = test_label.to(device)
            test_total += test_label.shape[0]
            mask = test_data["attention_mask"].squeeze(1).to(device)
            input_id = test_data["input_ids"].squeeze(1).to(device)
            loss, logits = model(input_id, mask, test_label)
            for i in range(logits.shape[0]):
                # remove pad tokens
                logits_clean = logits[i][test_label[i] != -100]
                label_clean = test_label[i][test_label[i] != -100]
                # calcluate acc and store prediciton and true labels
                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_test += acc.item()
                test_y_pred.append([ids_to_labels[x.item()] for x in predictions])
                test_y_true.append([ids_to_labels[x.item()] for x in label_clean])
    # metric summary
    summary = classification_report(y_true=test_y_true, y_pred=test_y_pred, zero_division=0)
    print(summary)
