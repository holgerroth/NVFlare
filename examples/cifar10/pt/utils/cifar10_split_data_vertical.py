# Copyright (c) 2022, NVIDIA CORPORATION.
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

from cifar10_vertical_data_splitter import Cifar10VerticalDataSplitter
from nvflare.apis.fl_context import FLContext
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from nvflare.apis.fl_constant import ReservedKey


def main():
    splitter = Cifar10VerticalDataSplitter(
        split_dir="/tmp/cifar10_vert_splits",
        overlap=10_000
    )

    # set up a dummy context for logging
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "local")
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "_")

    splitter.split(fl_ctx)  # will download to CIFAR10_ROOT defined in
    # Cifar10DataSplitter


if __name__ == "__main__":
    main()
