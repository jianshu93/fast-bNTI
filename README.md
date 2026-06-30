# Fast $\beta$ NTI computation at large scale

Fast exact $\beta$ NTI at scale for microbial communities.

`betanti` reads a dense text table by default, or a BIOM HDF5 table with `--biom`, plus a rooted Newick tree. It computes observed abundance-weighted betaMNTD, then computes betaNTI using taxa-label permutations without writing permutation matrices to disk.

The crate also builds `rc_bray`, a second binary for abundance-weighted Raup-Crick Bray-Curtis using the Stegen/Ning-style abundance null model.

The fast path is a batched tree distance transform: for each target-sample block, all present target tips are marked as zero and two tree passes compute distance from every node to the nearest target tip. Source samples are sparse relative-abundance vectors, so directed betaMNTD is a sparse dot product against those distance fields.

## Build

```bash
cargo build --release
```

## Usage

```bash
betanti \
  --tree tree.nwk \
  --biom table.biom \
  --permutations 999 \
  --threads 16 \
  --output betanti.tsv
```

Dense text input is the default table path:

```bash
betanti --tree tree.nwk --input table.tsv --output betanti.tsv
```

Use `--succ` to use the `succparen` balanced-parentheses tree backend. This keeps the default flat-array backend unchanged, but avoids storing the simple parent/preorder/postorder arrays in the betaMNTD transform:

```bash
betanti --succ --tree tree.nwk --biom table.biom --output betanti.tsv --succ
```

Text tables should have feature IDs in the first column and sample IDs in the remaining columns. Tab-delimited and comma-delimited files are autodetected from the header.

## Output

The default output is long-form TSV with one row per sample pair:

```text
sample1 sample2 beta_mntd null_mean null_sd beta_nti permutations
```

Use `--matrix-output` to additionally write a square betaNTI matrix.

## RC Bray

`rc_bray` accepts the same dense text-table orientation as `RC_bray_multip.r`: feature IDs in the first column and sample IDs in the remaining columns. BIOM HDF5 input is also supported.

```bash
rc_bray \
  --input table.tsv \
  --permutations 1000 \
  --threads 16 \
  --output rc_bray.tsv
```

R-script-compatible aliases are accepted:

```bash
rc_bray --input_file table.tsv --output_file rc_bray.tsv --processors 16
```

The implementation avoids dense randomized community matrices and the full 3D array of random Bray-Curtis matrices. It keeps samples sparse, preserves each sample's richness and total count under the null model, compares random and observed Bray-Curtis distances through exact integer shared-abundance counts, and updates the RC tally online.


## References
