# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

FROM python:3.12-slim
ARG FUNNEL_VERSION=0.11.14
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /tmp/funnel && cd /tmp/funnel \
    && curl -fsSLO "https://github.com/calypr/funnel/releases/download/v${FUNNEL_VERSION}/funnel-linux-amd64-v${FUNNEL_VERSION}.tar.gz" \
    && curl -fsSLO "https://github.com/calypr/funnel/releases/download/v${FUNNEL_VERSION}/funnel-v${FUNNEL_VERSION}-checksums.txt" \
    && grep " funnel-linux-amd64-v${FUNNEL_VERSION}.tar.gz$" "funnel-v${FUNNEL_VERSION}-checksums.txt" | sha256sum -c - \
    && tar -xzf "funnel-linux-amd64-v${FUNNEL_VERSION}.tar.gz" \
    && install funnel /usr/local/bin/funnel && rm -rf /tmp/funnel
COPY integration/ga4gh/requirements*.txt /opt/validation/integration/ga4gh/
RUN python -m pip install --no-cache-dir -r /opt/validation/integration/ga4gh/requirements-validation.txt
COPY integration/ga4gh /opt/validation/integration/ga4gh
COPY examples/hello-world/hello-numpy/client.py /opt/validation/examples/hello-world/hello-numpy/client.py
RUN python -m pip install --no-cache-dir --no-deps -e /opt/validation/integration/ga4gh
WORKDIR /opt/validation/integration/ga4gh
ENV MPLCONFIGDIR=/tmp/matplotlib
ENTRYPOINT ["python", "validation/funnel_run.py"]
