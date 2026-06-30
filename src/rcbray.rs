use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result};
use rand::{
    Rng, SeedableRng,
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
};
use rand_distr::Binomial;
use rayon::prelude::*;

use crate::table::{CountEntry, CountSample, read_biom_count_samples, read_text_count_samples};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RcBrayInputFormat {
    Text,
    Biom,
}

impl RcBrayInputFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Biom => "BIOM",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RcBrayOptions {
    pub permutations: usize,
    pub seed: u64,
}

pub fn read_rc_bray_samples(path: &Path, format: RcBrayInputFormat) -> Result<Vec<CountSample>> {
    match format {
        RcBrayInputFormat::Text => read_text_count_samples(path),
        RcBrayInputFormat::Biom => read_biom_count_samples(path),
    }
    .with_context(|| format!("read {} count table {}", format.label(), path.display()))
}

pub fn compute_rc_bray(samples: &[CountSample], opts: &RcBrayOptions) -> Result<Vec<f64>> {
    if samples.len() < 2 {
        anyhow::bail!("fewer than 2 non-empty samples after filtering");
    }
    if opts.permutations == 0 {
        anyhow::bail!("RC Bray requires at least one permutation");
    }

    let summary = CommunitySummary::new(samples)?;
    let pairs = sample_pairs(samples.len());
    let obs_shared = pairs
        .par_iter()
        .map(|&(i, j)| shared_min(&samples[i].entries, &samples[j].entries))
        .collect::<Vec<_>>();

    let mut less = vec![0usize; pairs.len()];
    let mut equal = vec![0usize; pairs.len()];
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut support_stamp = vec![0u64; summary.taxa_count];
    let mut stamp = 1u64;

    for _ in 0..opts.permutations {
        let randomized = randomize_samples(&summary, &mut rng, &mut support_stamp, &mut stamp)?;
        pairs
            .par_iter()
            .zip(obs_shared.par_iter())
            .zip(less.par_iter_mut())
            .zip(equal.par_iter_mut())
            .for_each(|(((&(i, j), &obs), lt), eq)| {
                let rand_shared = shared_min(&randomized[i], &randomized[j]);
                if rand_shared > obs {
                    *lt += 1;
                } else if rand_shared == obs {
                    *eq += 1;
                }
            });
    }

    let denom = opts.permutations as f64;
    Ok(less
        .into_iter()
        .zip(equal)
        .map(|(lt, eq)| {
            let alpha = (lt as f64 + 0.5 * eq as f64) / denom;
            2.0 * (alpha - 0.5)
        })
        .collect())
}

pub fn write_rc_matrix(path: &Path, samples: &[CountSample], rc: &[f64]) -> Result<()> {
    let n = samples.len();
    let expected = n * (n - 1) / 2;
    if rc.len() != expected {
        anyhow::bail!(
            "RC vector length {} does not match sample pair count {}",
            rc.len(),
            expected
        );
    }

    let mut mat = vec![0.0f64; n * n];
    let mut idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            mat[i * n + j] = rc[idx];
            mat[j * n + i] = rc[idx];
            idx += 1;
        }
    }

    let mut out = BufWriter::with_capacity(16 << 20, File::create(path)?);
    for sample in samples {
        out.write_all(b"\t")?;
        out.write_all(sample.name.as_bytes())?;
    }
    out.write_all(b"\n")?;

    let mut buf = ryu::Buffer::new();
    for i in 0..n {
        out.write_all(samples[i].name.as_bytes())?;
        for j in 0..n {
            out.write_all(b"\t")?;
            out.write_all(buf.format_finite(mat[i * n + j]).as_bytes())?;
        }
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn sample_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

fn shared_min(a: &[CountEntry], b: &[CountEntry]) -> u64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut shared = 0u64;
    while i < a.len() && j < b.len() {
        let ai = a[i];
        let bj = b[j];
        if ai.taxon == bj.taxon {
            shared += ai.count.min(bj.count);
            i += 1;
            j += 1;
        } else if ai.taxon < bj.taxon {
            i += 1;
        } else {
            j += 1;
        }
    }
    shared
}

fn randomize_samples<R: Rng + ?Sized>(
    summary: &CommunitySummary,
    rng: &mut R,
    support_stamp: &mut [u64],
    stamp: &mut u64,
) -> Result<Vec<Vec<CountEntry>>> {
    let mut out = Vec::with_capacity(summary.samples.len());
    for spec in &summary.samples {
        let support = draw_support(summary, spec.richness, rng, support_stamp, stamp)?;
        out.push(draw_counts(summary, spec.total, support, rng)?);
    }
    Ok(out)
}

fn draw_support<R: Rng + ?Sized>(
    summary: &CommunitySummary,
    richness: usize,
    rng: &mut R,
    support_stamp: &mut [u64],
    stamp: &mut u64,
) -> Result<Vec<usize>> {
    if richness == 0 {
        return Ok(Vec::new());
    }
    if richness > summary.positive_taxa {
        anyhow::bail!(
            "sample richness {richness} exceeds positive taxon count {}",
            summary.positive_taxa
        );
    }
    if richness == summary.positive_taxa {
        return Ok(summary.positive_taxa_ids.clone());
    }

    let current = *stamp;
    *stamp = stamp
        .checked_add(1)
        .context("support sampling stamp counter overflow")?;
    let mut support = Vec::with_capacity(richness);
    while support.len() < richness {
        let taxon = summary.occupancy_sampler.sample(rng);
        if support_stamp[taxon] != current {
            support_stamp[taxon] = current;
            support.push(taxon);
        }
    }
    Ok(support)
}

fn draw_counts<R: Rng + ?Sized>(
    summary: &CommunitySummary,
    total: u64,
    mut support: Vec<usize>,
    rng: &mut R,
) -> Result<Vec<CountEntry>> {
    let richness = support.len();
    if richness == 0 {
        return Ok(Vec::new());
    }
    if total < richness as u64 {
        anyhow::bail!("sample total {total} is smaller than richness {richness}");
    }
    support.sort_unstable();

    let mut counts = vec![1u64; richness];
    let mut remaining = total - richness as u64;
    if remaining > 0 {
        let mut remaining_weight = support
            .iter()
            .map(|&taxon| summary.abundance_weights[taxon])
            .sum::<f64>();

        for k in 0..richness - 1 {
            if remaining == 0 {
                break;
            }
            let weight = summary.abundance_weights[support[k]];
            let p = if remaining_weight <= 0.0 {
                0.0
            } else {
                (weight / remaining_weight).clamp(0.0, 1.0)
            };
            let extra = draw_binomial(remaining, p, rng)?;
            counts[k] += extra;
            remaining -= extra;
            remaining_weight -= weight;
        }
        counts[richness - 1] += remaining;
    }

    Ok(support
        .into_iter()
        .zip(counts)
        .map(|(taxon, count)| CountEntry { taxon, count })
        .collect())
}

fn draw_binomial<R: Rng + ?Sized>(n: u64, p: f64, rng: &mut R) -> Result<u64> {
    if n == 0 || p <= 0.0 {
        return Ok(0);
    }
    if p >= 1.0 {
        return Ok(n);
    }
    let dist = Binomial::new(n, p).context("construct binomial distribution")?;
    Ok(dist.sample(rng))
}

struct CommunitySummary {
    samples: Vec<SampleSpec>,
    abundance_weights: Vec<f64>,
    occupancy_sampler: WeightedIndex<f64>,
    positive_taxa_ids: Vec<usize>,
    positive_taxa: usize,
    taxa_count: usize,
}

struct SampleSpec {
    richness: usize,
    total: u64,
}

impl CommunitySummary {
    fn new(samples: &[CountSample]) -> Result<Self> {
        let taxa_count = samples
            .iter()
            .flat_map(|sample| sample.entries.iter().map(|entry| entry.taxon))
            .max()
            .map(|v| v + 1)
            .context("no taxa with positive counts")?;

        let mut occupancy_weights = vec![0.0f64; taxa_count];
        let mut abundance_weights = vec![0.0f64; taxa_count];
        for sample in samples {
            for entry in &sample.entries {
                occupancy_weights[entry.taxon] += 1.0;
                abundance_weights[entry.taxon] += entry.count as f64;
            }
        }

        let positive_taxa_ids = abundance_weights
            .iter()
            .enumerate()
            .filter_map(|(taxon, &weight)| (weight > 0.0).then_some(taxon))
            .collect::<Vec<_>>();
        let positive_taxa = positive_taxa_ids.len();
        if positive_taxa == 0 {
            anyhow::bail!("no taxa with positive counts");
        }

        let occupancy_sampler = WeightedIndex::new(occupancy_weights)
            .context("construct occupancy weighted sampler")?;
        let samples = samples
            .iter()
            .map(|sample| SampleSpec {
                richness: sample.entries.len(),
                total: sample.total,
            })
            .collect();

        Ok(Self {
            samples,
            abundance_weights,
            occupancy_sampler,
            positive_taxa_ids,
            positive_taxa,
            taxa_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::shared_min;
    use crate::table::CountEntry;

    #[test]
    fn shared_min_uses_sparse_intersection() {
        let a = vec![
            CountEntry { taxon: 0, count: 3 },
            CountEntry { taxon: 4, count: 2 },
            CountEntry { taxon: 5, count: 9 },
        ];
        let b = vec![
            CountEntry { taxon: 1, count: 8 },
            CountEntry { taxon: 4, count: 7 },
            CountEntry { taxon: 5, count: 1 },
        ];
        assert_eq!(shared_min(&a, &b), 3);
    }
}
