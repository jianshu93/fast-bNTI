use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgGroup, Command};
use log::{info, warn};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use betanti::nmtd::{BetaOptions, compute_pair_values, write_long_output, write_matrix_output};
use betanti::table::{WeightMode, read_biom_samples, read_text_samples};
use betanti::tree::PhyloTree;

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();

    let m = Command::new("betanti")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Exact fast betaMNTD and betaNTI from BIOM/TSV tables and Newick trees")
        .arg(
            Arg::new("tree")
                .short('t')
                .long("tree")
                .help("Input tree in Newick format")
                .required(true)
                .value_name("TREE")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Dense feature table in text format. First column is feature ID; remaining columns are samples")
                .value_name("INPUT")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("biom")
                .short('b')
                .long("biom")
                .help("Feature table in BIOM HDF5 format")
                .value_name("BIOM")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output long-form betaNTI TSV")
                .value_name("OUTPUT")
                .default_value("betanti.tsv")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("matrix-output")
                .long("matrix-output")
                .help("Optional square betaNTI matrix output path")
                .value_name("MATRIX_OUTPUT")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("permutations")
                .short('p')
                .long("permutations")
                .help("Number of taxa-label permutations for the null model")
                .value_name("PERMUTATIONS")
                .default_value("999")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("threads")
                .short('T')
                .long("threads")
                .help("Number of threads. Defaults to all logical cores")
                .value_name("THREADS")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("block-size")
                .long("block-size")
                .help("Number of target samples processed per tree-transform block")
                .value_name("BLOCK_SIZE")
                .default_value("8")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Random seed for taxa-label permutations")
                .value_name("SEED")
                .default_value("1337")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("weight-mode")
                .long("weight-mode")
                .help("How table values are converted into per-sample weights")
                .value_name("WEIGHT_MODE")
                .default_value("relative")
                .value_parser(["relative", "presence"]),
        )
        .arg(
            Arg::new("succ")
                .long("succ")
                .help("Use the succparen balanced-parentheses tree backend instead of the simple flat-array backend")
                .action(ArgAction::SetTrue),
        )
        .group(
            ArgGroup::new("table")
                .args(["input", "biom"])
                .required(true)
                .multiple(false),
        )
        .get_matches();

    let tree_path = m.get_one::<PathBuf>("tree").unwrap();
    let input_path = m.get_one::<PathBuf>("input");
    let biom_path = m.get_one::<PathBuf>("biom");
    let output_path = m.get_one::<PathBuf>("output").unwrap();
    let matrix_output_path = m.get_one::<PathBuf>("matrix-output");
    let permutations = *m.get_one::<usize>("permutations").unwrap();
    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(num_cpus::get)
        .max(1);
    let block_size = *m.get_one::<usize>("block-size").unwrap();
    let seed = *m.get_one::<u64>("seed").unwrap();
    let weight_mode = parse_weight_mode(m.get_one::<String>("weight-mode").unwrap().as_str())?;
    let succ = m.get_flag("succ");

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("initialize rayon thread pool")?;
    info!("using {} rayon threads", rayon::current_num_threads());

    let t0 = Instant::now();
    let tree = if succ {
        info!("loading tree with succparen balanced-parentheses representation");
        PhyloTree::from_newick_path_succ(tree_path)
    } else {
        info!("loading tree with simple Newick traversal");
        PhyloTree::from_newick_path(tree_path)
    }
    .with_context(|| format!("load tree {}", tree_path.display()))?;
    info!(
        "loaded tree: {} nodes, {} tips in {} ms",
        tree.node_count(),
        tree.leaf_count(),
        t0.elapsed().as_millis()
    );

    let t1 = Instant::now();
    let samples = if let Some(path) = input_path {
        read_text_samples(path, &tree, weight_mode)
            .with_context(|| format!("read text table {}", path.display()))?
    } else {
        let path = biom_path.expect("BIOM path required");
        read_biom_samples(path, &tree, weight_mode)
            .with_context(|| format!("read BIOM table {}", path.display()))?
    };

    if samples.len() < 2 {
        anyhow::bail!("fewer than 2 non-empty samples after filtering");
    }
    let total_nnz: usize = samples.iter().map(|s| s.entries.len()).sum();
    info!(
        "loaded {} non-empty samples, {} total nonzero sample-taxa entries in {} ms",
        samples.len(),
        total_nnz,
        t1.elapsed().as_millis()
    );

    let opts = BetaOptions {
        block_size: block_size.max(1),
    };

    let t2 = Instant::now();
    info!("computing observed betaMNTD");
    let observed = compute_pair_values(&tree, &samples, None, &opts);
    info!("observed betaMNTD done in {} ms", t2.elapsed().as_millis());

    let n_pairs = observed.len();
    let mut null_mean = vec![0.0f64; n_pairs];
    let mut null_m2 = vec![0.0f64; n_pairs];

    if permutations > 0 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut perm: Vec<usize> = (0..tree.leaf_count()).collect();
        let t3 = Instant::now();

        for p in 1..=permutations {
            perm.shuffle(&mut rng);
            let vals = compute_pair_values(&tree, &samples, Some(&perm), &opts);
            update_welford(&mut null_mean, &mut null_m2, &vals, p);

            if p == 1 || p == permutations || p % 10 == 0 {
                info!(
                    "permutation {}/{} complete (elapsed {} ms)",
                    p,
                    permutations,
                    t3.elapsed().as_millis()
                );
            }
        }
    } else {
        warn!("--permutations 0: beta_nti/null columns will be NaN");
        null_mean.fill(f64::NAN);
    }

    let (null_mean, null_sd, beta_nti) =
        finalize_beta_nti_cpu(observed.as_slice(), null_mean, &null_m2, permutations);

    write_long_output(
        output_path,
        &samples,
        &observed,
        &null_mean,
        &null_sd,
        &beta_nti,
        permutations,
    )
    .with_context(|| format!("write output {}", output_path.display()))?;
    info!("wrote {}", output_path.display());

    if let Some(path) = matrix_output_path {
        write_matrix_output(path, &samples, &beta_nti)
            .with_context(|| format!("write matrix output {}", path.display()))?;
        info!("wrote {}", path.display());
    }

    Ok(())
}

fn finalize_beta_nti_cpu(
    observed: &[f64],
    null_mean: Vec<f64>,
    null_m2: &[f64],
    permutations: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let null_sd = finalize_sd(null_m2, permutations);
    let beta_nti = observed
        .iter()
        .zip(null_mean.iter())
        .zip(null_sd.iter())
        .map(|((&obs, &mean), &sd)| {
            if permutations < 2 || sd == 0.0 || !sd.is_finite() {
                f64::NAN
            } else {
                (obs - mean) / sd
            }
        })
        .collect::<Vec<_>>();
    (null_mean, null_sd, beta_nti)
}

fn parse_weight_mode(value: &str) -> Result<WeightMode> {
    match value {
        "relative" => Ok(WeightMode::Relative),
        "presence" => Ok(WeightMode::Presence),
        _ => anyhow::bail!("unsupported weight mode: {value}"),
    }
}

fn update_welford(mean: &mut [f64], m2: &mut [f64], values: &[f64], count: usize) {
    let n = count as f64;
    for ((mu, ss), &x) in mean.iter_mut().zip(m2.iter_mut()).zip(values.iter()) {
        let delta = x - *mu;
        *mu += delta / n;
        let delta2 = x - *mu;
        *ss += delta * delta2;
    }
}

fn finalize_sd(m2: &[f64], permutations: usize) -> Vec<f64> {
    if permutations < 2 {
        return vec![f64::NAN; m2.len()];
    }
    let denom = (permutations - 1) as f64;
    m2.iter().map(|v| (v / denom).sqrt()).collect()
}
