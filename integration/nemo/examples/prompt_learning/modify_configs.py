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
import glob
import json
import os


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def save_config(config_file, config):
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_folder",
        type=str,
        help="Folder containing job config files in JSON format.",
    )
    parser.add_argument(
        "--root_dir", type=str, help="Root folder containing the example with data and models.", default=os.getcwd()
    )
    args = parser.parse_args()

    # modify client configs
    client_cfg_files = glob.glob(os.path.join(args.job_folder, "**", "config_fed_client.json"), recursive=True)
    assert len(client_cfg_files), f"No client configs found in {args.job_folder}"
    for client_cfg_file in client_cfg_files:
        client_cfg = load_config(client_cfg_file)
        client_cfg["ROOT_DIR"] = args.root_dir

        save_config(client_cfg_file, client_cfg)

    # modify server config
    server_cfg_file = glob.glob(os.path.join(args.job_folder, "**", "config_fed_server.json"), recursive=True)
    assert len(server_cfg_file), f"Expected one server config file in {args.job_folder}"
    server_cfg_file = server_cfg_file[0]
    server_cfg = load_config(server_cfg_file)
    server_cfg["ROOT_DIR"] = args.root_dir

    save_config(server_cfg_file, server_cfg)

    print(f"Set ROOT_DIR to {args.root_dir}")


if __name__ == "__main__":
    main()
