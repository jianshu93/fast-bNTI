use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result};
use hdf5::{File as H5File, types::VarLenUnicode};
use log::{info, warn};

use crate::tree::PhyloTree;

#[derive(Copy, Clone, Debug)]
pub enum WeightMode {
    Relative,
    Presence,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub name: String,
    pub entries: Vec<SampleEntry>,
}

#[derive(Copy, Clone, Debug)]
pub struct SampleEntry {
    pub leaf_ord: usize,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct CountSample {
    pub name: String,
    pub entries: Vec<CountEntry>,
    pub total: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct CountEntry {
    pub taxon: usize,
    pub count: u64,
}

pub fn read_text_samples(path: &Path, tree: &PhyloTree, mode: WeightMode) -> Result<Vec<Sample>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().context("empty text table")??;
    let delimiter = detect_delimiter(&header);
    let header_fields = split_fields(&header, delimiter);
    if header_fields.len() < 2 {
        anyhow::bail!("table header must contain one feature column and at least one sample");
    }

    let sample_names = header_fields[1..]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let nsamp = sample_names.len();
    let mut raw_entries = vec![Vec::<(usize, f64)>::new(); nsamp];
    let mut missing_taxa = 0usize;
    let mut parsed_taxa = 0usize;

    for (line_no, line) in lines.enumerate() {
        let row = line?;
        if row.trim().is_empty() {
            continue;
        }
        let fields = split_fields(&row, delimiter);
        if fields.is_empty() {
            continue;
        }
        parsed_taxa += 1;
        let Some(&leaf_ord) = tree.tip_to_leaf_ord().get(fields[0]) else {
            missing_taxa += 1;
            continue;
        };

        for s in 0..nsamp {
            let val = fields
                .get(s + 1)
                .map(|v| v.trim().parse::<f64>().unwrap_or(0.0))
                .unwrap_or(0.0);
            if val > 0.0 {
                raw_entries[s].push((leaf_ord, val));
            } else if val < 0.0 {
                anyhow::bail!(
                    "negative table value at line {}, sample {}",
                    line_no + 2,
                    sample_names[s]
                );
            }
        }
    }

    log_mapping_summary(parsed_taxa, missing_taxa);
    Ok(normalize_samples(sample_names, raw_entries, mode))
}

pub fn read_biom_samples(path: &Path, tree: &PhyloTree, mode: WeightMode) -> Result<Vec<Sample>> {
    let f = H5File::open(path).with_context(|| format!("open BIOM file {}", path.display()))?;

    let taxa = read_utf8(&f, "observation/ids").context("missing observation/ids")?;
    let samples = read_utf8(&f, "sample/ids").context("missing sample/ids")?;
    let indptr = try_usize_paths(&f, "observation", "indptr")?;
    let indices = try_usize_paths(&f, "observation", "indices")?;
    let data = try_f64_paths(&f, "observation", "data")?;

    if indptr.len() != taxa.len() + 1 {
        anyhow::bail!(
            "BIOM observation indptr length {} does not match taxa length {} + 1",
            indptr.len(),
            taxa.len()
        );
    }
    if indices.len() != data.len() {
        anyhow::bail!(
            "BIOM indices length {} does not match data length {}",
            indices.len(),
            data.len()
        );
    }

    let nsamp = samples.len();
    let mut raw_entries = vec![Vec::<(usize, f64)>::new(); nsamp];
    let mut missing_taxa = 0usize;

    for (r, taxon) in taxa.iter().enumerate() {
        let Some(&leaf_ord) = tree.tip_to_leaf_ord().get(taxon.as_str()) else {
            missing_taxa += 1;
            continue;
        };

        for k in indptr[r]..indptr[r + 1] {
            let s = indices[k];
            if s >= nsamp {
                anyhow::bail!("BIOM sample index {s} exceeds sample count {nsamp}");
            }
            let val = data[k];
            if val > 0.0 {
                raw_entries[s].push((leaf_ord, val));
            } else if val < 0.0 {
                anyhow::bail!("negative BIOM value at observation row {r}, data index {k}");
            }
        }
    }

    log_mapping_summary(taxa.len(), missing_taxa);
    Ok(normalize_samples(samples, raw_entries, mode))
}

pub fn read_text_count_samples(path: &Path) -> Result<Vec<CountSample>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().context("empty text table")??;
    let delimiter = detect_delimiter(&header);
    let header_fields = split_fields(&header, delimiter);
    if header_fields.len() < 2 {
        anyhow::bail!("table header must contain one feature column and at least one sample");
    }

    let sample_names = header_fields[1..]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let nsamp = sample_names.len();
    let mut raw_entries = vec![Vec::<CountEntry>::new(); nsamp];
    let mut taxa_seen = 0usize;

    for (line_no, line) in lines.enumerate() {
        let row = line?;
        if row.trim().is_empty() {
            continue;
        }
        let fields = split_fields(&row, delimiter);
        if fields.is_empty() {
            continue;
        }
        let taxon = taxa_seen;
        taxa_seen += 1;

        for s in 0..nsamp {
            let count = fields
                .get(s + 1)
                .map(|v| parse_count_value(v.trim(), line_no + 2, &sample_names[s]))
                .transpose()?
                .unwrap_or(0);
            if count > 0 {
                raw_entries[s].push(CountEntry { taxon, count });
            }
        }
    }

    Ok(normalize_count_samples(sample_names, raw_entries))
}

pub fn read_biom_count_samples(path: &Path) -> Result<Vec<CountSample>> {
    let f = H5File::open(path).with_context(|| format!("open BIOM file {}", path.display()))?;

    let taxa = read_utf8(&f, "observation/ids").context("missing observation/ids")?;
    let samples = read_utf8(&f, "sample/ids").context("missing sample/ids")?;
    let indptr = try_usize_paths(&f, "observation", "indptr")?;
    let indices = try_usize_paths(&f, "observation", "indices")?;
    let data = try_f64_paths(&f, "observation", "data")?;

    if indptr.len() != taxa.len() + 1 {
        anyhow::bail!(
            "BIOM observation indptr length {} does not match taxa length {} + 1",
            indptr.len(),
            taxa.len()
        );
    }
    if indices.len() != data.len() {
        anyhow::bail!(
            "BIOM indices length {} does not match data length {}",
            indices.len(),
            data.len()
        );
    }

    let nsamp = samples.len();
    let mut raw_entries = vec![Vec::<CountEntry>::new(); nsamp];
    for r in 0..taxa.len() {
        for k in indptr[r]..indptr[r + 1] {
            let s = indices[k];
            if s >= nsamp {
                anyhow::bail!("BIOM sample index {s} exceeds sample count {nsamp}");
            }
            let count = f64_to_count(data[k])
                .with_context(|| format!("BIOM observation row {r}, data index {k}"))?;
            if count > 0 {
                raw_entries[s].push(CountEntry { taxon: r, count });
            }
        }
    }

    Ok(normalize_count_samples(samples, raw_entries))
}

fn normalize_samples(
    names: Vec<String>,
    raw_entries: Vec<Vec<(usize, f64)>>,
    mode: WeightMode,
) -> Vec<Sample> {
    names
        .into_iter()
        .zip(raw_entries)
        .filter_map(|(name, entries)| {
            if entries.is_empty() {
                warn!("dropping empty sample {name}");
                return None;
            }

            let normalized = match mode {
                WeightMode::Relative => {
                    let total = entries.iter().map(|(_, v)| *v).sum::<f64>();
                    if total <= 0.0 {
                        warn!("dropping zero-sum sample {name}");
                        return None;
                    }
                    entries
                        .into_iter()
                        .map(|(leaf_ord, v)| SampleEntry {
                            leaf_ord,
                            weight: (v / total) as f32,
                        })
                        .collect()
                }
                WeightMode::Presence => {
                    let w = 1.0f32 / entries.len() as f32;
                    entries
                        .into_iter()
                        .map(|(leaf_ord, _)| SampleEntry {
                            leaf_ord,
                            weight: w,
                        })
                        .collect()
                }
            };

            Some(Sample {
                name,
                entries: normalized,
            })
        })
        .collect()
}

fn normalize_count_samples(
    names: Vec<String>,
    raw_entries: Vec<Vec<CountEntry>>,
) -> Vec<CountSample> {
    names
        .into_iter()
        .zip(raw_entries)
        .filter_map(|(name, mut entries)| {
            if entries.is_empty() {
                warn!("dropping empty sample {name}");
                return None;
            }

            entries.sort_unstable_by_key(|entry| entry.taxon);
            let total = entries.iter().map(|entry| entry.count).sum::<u64>();
            if total == 0 {
                warn!("dropping zero-sum sample {name}");
                return None;
            }
            Some(CountSample {
                name,
                entries,
                total,
            })
        })
        .collect()
}

fn detect_delimiter(header: &str) -> char {
    if header.contains('\t') {
        '\t'
    } else if header.contains(',') {
        ','
    } else {
        '\t'
    }
}

fn split_fields(line: &str, delimiter: char) -> Vec<&str> {
    line.trim_end_matches(['\r', '\n'])
        .split(delimiter)
        .collect()
}

fn parse_count_value(raw: &str, line_no: usize, sample: &str) -> Result<u64> {
    if raw.is_empty() {
        return Ok(0);
    }
    let value = raw
        .parse::<f64>()
        .with_context(|| format!("parse count value at line {line_no}, sample {sample}: {raw}"))?;
    f64_to_count(value).with_context(|| format!("line {line_no}, sample {sample}"))
}

fn f64_to_count(value: f64) -> Result<u64> {
    if !value.is_finite() {
        anyhow::bail!("count is not finite: {value}");
    }
    if value < 0.0 {
        anyhow::bail!("negative count: {value}");
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        anyhow::bail!("RC Bray requires integer counts, got {value}");
    }
    if rounded > u64::MAX as f64 {
        anyhow::bail!("count exceeds u64 range: {value}");
    }
    Ok(rounded as u64)
}

fn log_mapping_summary(total_taxa: usize, missing_taxa: usize) {
    if missing_taxa > 0 {
        warn!(
            "{} of {} table taxa were not found in the tree and were ignored",
            missing_taxa, total_taxa
        );
    }
    info!(
        "{} of {} table taxa mapped to tree tips",
        total_taxa.saturating_sub(missing_taxa),
        total_taxa
    );
}

fn read_utf8(f: &H5File, path: &str) -> Result<Vec<String>> {
    Ok(f.dataset(path)?
        .read_1d::<VarLenUnicode>()?
        .into_iter()
        .map(|v| v.as_str().to_owned())
        .collect())
}

fn try_usize_paths(f: &H5File, group: &str, name: &str) -> Result<Vec<usize>> {
    read_usize_flex(f, &format!("{group}/matrix/{name}"))
        .or_else(|_| read_usize_flex(f, &format!("{group}/{name}")))
        .with_context(|| format!("missing {group}/**/{name}"))
}

fn try_f64_paths(f: &H5File, group: &str, name: &str) -> Result<Vec<f64>> {
    read_f64_flex(f, &format!("{group}/matrix/{name}"))
        .or_else(|_| read_f64_flex(f, &format!("{group}/{name}")))
        .with_context(|| format!("missing {group}/**/{name}"))
}

fn read_usize_flex(f: &H5File, path: &str) -> Result<Vec<usize>> {
    let ds = f.dataset(path)?;
    if let Ok(v) = ds.read_raw::<usize>() {
        return Ok(v);
    }
    if let Ok(v) = ds.read_raw::<u64>() {
        return Ok(v.into_iter().map(|x| x as usize).collect());
    }
    if let Ok(v) = ds.read_raw::<i64>() {
        return Ok(v.into_iter().map(|x| x as usize).collect());
    }
    if let Ok(v) = ds.read_raw::<u32>() {
        return Ok(v.into_iter().map(|x| x as usize).collect());
    }
    if let Ok(v) = ds.read_raw::<i32>() {
        return Ok(v.into_iter().map(|x| x as usize).collect());
    }
    anyhow::bail!("could not read integer dataset {path}")
}

fn read_f64_flex(f: &H5File, path: &str) -> Result<Vec<f64>> {
    let ds = f.dataset(path)?;
    if let Ok(v) = ds.read_raw::<f64>() {
        return Ok(v);
    }
    if let Ok(v) = ds.read_raw::<f32>() {
        return Ok(v.into_iter().map(|x| x as f64).collect());
    }
    if let Ok(v) = ds.read_raw::<u64>() {
        return Ok(v.into_iter().map(|x| x as f64).collect());
    }
    if let Ok(v) = ds.read_raw::<i64>() {
        return Ok(v.into_iter().map(|x| x as f64).collect());
    }
    if let Ok(v) = ds.read_raw::<u32>() {
        return Ok(v.into_iter().map(|x| x as f64).collect());
    }
    if let Ok(v) = ds.read_raw::<i32>() {
        return Ok(v.into_iter().map(|x| x as f64).collect());
    }
    anyhow::bail!("could not read numeric dataset {path}")
}
