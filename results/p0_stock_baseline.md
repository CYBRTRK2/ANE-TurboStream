| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_M - 2.7 bpw |  10.60 GiB |    34.66 B | MTL,BLAS   |       4 |            pp64 |         59.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_M - 2.7 bpw |  10.60 GiB |    34.66 B | MTL,BLAS   |       4 |            tg32 |         22.65 ± 0.00 |

build: da3b409e (8448)
