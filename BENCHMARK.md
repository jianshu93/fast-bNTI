# Benchmark Notes

Benchmarks below were run on the added GWMC data folder with 16 logical CPUs available.

Input files:

- `data/GWMC_16S_otutab.biom`
- `data/GWMC_rep_seqs_all.tre`

The 10% and 1% text subsets were generated from the BIOM sample matrix for testing text input and R comparison.

## Rust

| input | samples | taxa union | permutations | wall time |
|---|---:|---:|---:|---:|
| text subset | 12 | 9,732 | 10 | 0.41 s |
| text subset | 119 | 30,185 | 10 | 0.56 s |
| full BIOM | 1,186 | 96,148 | 10 | 6.02 s |

Observed-only final-folder verification on the full BIOM with 4 threads took 1.55 s and wrote 702,705 sample-pair rows plus header.

## R / picante

The provided R workflow uses `picante::comdistnt` with a dense `cophenetic` matrix and `taxaShuffle`.

On the 1% text subset:

| step | elapsed |
|---|---:|
| setup/tree match/cophenetic | 1.797 s |
| observed betaMNTD | 15.169 s |
| 10 null permutations | 143.946 s |
| total | 160.912 s |

The Rust and R observed betaMNTD values agreed to floating-point precision on the same subset:

- 66 unique sample pairs
- max absolute difference: `5.062456148730021e-10`
- mean absolute difference: `1.4631121842803322e-10`

The 10% subset has 30,185 taxa after trimming. A dense R cophenetic matrix at that size is already several GB before overhead, so the R benchmark was kept to the 1% subset.
