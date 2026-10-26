# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cwlVersion: v1.2
class: CommandLineTool
label: Synthetic site calculation for the NVFLARE Client API example
baseCommand: [python3]
requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: input.json
        entry: $(inputs.document)
inputs:
  document: string
  script:
    type: File
    inputBinding:
      position: 1
arguments:
  - position: 2
    valueFrom: input.json
  - position: 3
    valueFrom: output.json
outputs:
  result:
    type: File
    outputBinding:
      glob: output.json
