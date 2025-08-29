# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List
from df_statistics import DFStatistics

from nvflare.job_config.stats_job import StatsJob
from nvflare.recipe.spec import Recipe


class FedStatsRecipe(Recipe):


    def __init__(
        self,
        *,
        name: str,
        data_path: str,
        stats_output_path: str,
        sites: List[str],
    ):
       
        output_path = stats_output_path

        statistic_configs = {
            "count": {},
            "mean": {},
            "sum": {},
            "stddev": {},
            "histogram": {"*": {"bins": 64}},
            #"histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
            #"quantile": {"*": [0.1, 0.5, 0.9], "Age": [0.1, 0.5, 0.9]},
        }
        # define local stats generator
        df_stats_generator = DFStatistics(data_path=data_path)

        job = StatsJob(
            job_name=name,
            statistic_configs=statistic_configs,
            stats_generator=df_stats_generator,
            output_path=output_path,
        )

        #sites = [f"site{i + 1}" for i in range(n_clients)]
        job.setup_clients(sites)

        Recipe.__init__(self, job)
