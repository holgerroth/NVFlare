# Upstream schema fixtures

These unmodified schemas are retained for offline contract validation. Their upstream license is Apache-2.0; see the integration's LICENSE.

| File | Source | Git blob SHA |
|---|---|---|
| `wes-1.1.0.yaml` | [WES release 1.1.0](https://github.com/ga4gh/workflow-execution-service-schemas/blob/1.1.0/openapi/workflow_execution_service.openapi.yaml) | `025b7d56a91a9700790abfce9a3f8645d842b195` |
| `service-info-1.0.0.yaml` | [Service Info v1.0.0](https://github.com/ga4gh-discovery/ga4gh-service-info/blob/v1.0.0/service-info.yaml) | See upstream version tag |
| `tes-1.1.0.yaml` | [TES develop schema](https://github.com/ga4gh/task-execution-schemas/blob/develop/openapi/task_execution_service.openapi.yaml), declaring version 1.1.0 | `0b3bb8e15b4460493341125d3f5f0c43aba2652c` |

The TES fixture is pinned by content; it is not described as a tagged TES release. Tests resolve WES's Service Info reference locally without network access. Schema validation checks the portions exercised by this pilot and does not certify arbitrary-workflow portability or production readiness.
