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

import os
import argparse
import glob
import pandas as pd


def data_args():
    parser = argparse.ArgumentParser(description="Turn csv to jsonl files for nemo")
    parser.add_argument("--in_dir", type=str, required=True, help="Input dir of csv files")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = data_args()
    
    if not args.out_dir:
        args.out_dir = args.in_dir
    
    file_list = glob.glob(os.path.join(args.in_dir, "**", "*.csv"), recursive=True)
    assert len(file_list) > 0, f"No files found in {args.in_dir}"
    
    print(f"Converting {len(file_list)} csv files to jsonl")
    # load training data
    data_combined = pd.DataFrame()
    for file in file_list:
        data = pd.read_csv(file)
        #data = data[["text", "labels"]]
        data = data.rename(columns={"text": "input", "labels": "output"})
    
        # save the combined data
        output_path = file.replace(args.in_dir, args.out_dir)
        output_path = output_path.replace(".csv", ".jsonl")
        assert not os.path.isfile(output_path), f"{output_path} already exists!"
        with open(output_path, "w") as f:
            f.write(data.to_json(orient="records", lines=True))
            print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
