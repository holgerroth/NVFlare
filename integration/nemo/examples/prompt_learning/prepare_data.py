# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import shutil
import glob

def load_jsonl(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str
    )
    parser.add_argument(
        "--out_dir",
        type=str
    ) 
    parser.add_argument(
        "--taskname",
        type=str,
        default="chat"
    )     
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.in_dir, "**", "*.jsonl"), recursive=True)
    assert len(files) > 0
    
    for i, file in enumerate(files):
        out_file = file.replace(args.in_dir, args.out_dir)
        
        out_dir = os.path.dirname(out_file)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        data = load_jsonl(file)
        for d in data:
            d["taskname"] = args.taskname
    
        save_jsonl(out_file, data)
        print(f"Saved file {i+1} of {len(files)} to {out_file}")
        

if __name__ == "__main__":
    main()
