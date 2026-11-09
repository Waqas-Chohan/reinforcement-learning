# SE4095 Assignment 03 Workplace Dataset

Extract this archive into the assignment project so the three CSV files and this README are directly inside `data/`. Use the supplied files without modification.

| File | Columns | Data rows |
|---|---|---:|
| nodes.csv | node_id, original_id | 92 |
| edges.csv | source, target | 755 |
| departments.csv | node_id, department | 92 |

Load every node from nodes.csv before adding edges. Use integer node_id values for graph construction and actions. Use original_id only as a provenance mapping. Join department labels by node_id for visualization and final analysis. Keep department labels out of policy inputs, rewards and tuning.

## Preparation

The source metadata contains 92 participants. Original numeric participant IDs were sorted and remapped to IDs 0–91. All 9,827 recorded contact rows in the official archive were considered. Contacts across the full observation period were aggregated into one undirected, unweighted edge per distinct pair. Self-loops were excluded. Edges are sorted numerically by source and target, with source < target. The resulting graph has 0 isolated participants and 5 recorded department labels. Timestamp and duration are excluded from the prepared graph.

The contacts were observed. The assignment's message spread and activation probability of 0.10 belong to the Independent Cascade simulation.

## Source and attribution

SocioPatterns, Contacts in a Workplace. Data collected June 24 to July 3, 2013. The original release is available under a Creative Commons Public Domain Dedication.

- Dataset and terms: https://sociopatterns.org/datasets/contacts-in-a-workplace/
- Raw contacts: https://sociopatterns.org/assets/data/workplace_InVS_tij.dat.zip
- Department metadata: https://sociopatterns.org/assets/data/workplace_InVS_metadata.txt
- Génois et al. (2015), Network Science, 3(3), 326–347. https://doi.org/10.1017/nws.2015.10

## CSV integrity

SHA-256 hashes for the distributed CSV files.

- `nodes.csv` — `22cbebc2094014e7b03257b11527c24804c2a63152ce3e9d7abc70f6fbe50076`
- `edges.csv` — `b413b8416469105d42e53eec557ff5c7bc3d30179a0c52a0c5cd8c402cb50438`
- `departments.csv` — `a2821b03da4dc96d887e5f39d2851130a87e41e94ad313bbef26f26b2927a7ee`
