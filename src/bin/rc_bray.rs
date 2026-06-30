use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgGroup, Command};
use log::info;

use betanti::rcbray::{
    RcBrayInputFormat, RcBrayOptions, compute_rc_bray, read_rc_bray_samples, write_rc_matrix,
};

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();

    let m = Command::new("rc_bray")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Fast abundance-weighted Raup-Crick metric using Bray-Curtis null distances")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .alias("input_file")
                .help("Dense count table in text format. First column is feature ID; remaining columns are samples")
                .value_name("INPUT")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("biom")
                .short('b')
                .long("biom")
                .help("Count table in BIOM HDF5 format")
                .value_name("BIOM")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .alias("output_file")
                .help("Output square RC Bray matrix TSV")
                .value_name("OUTPUT")
                .default_value("rc_bray.tsv")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("permutations")
                .short('p')
                .long("permutations")
                .help("Number of null randomizations")
                .value_name("PERMUTATIONS")
                .default_value("1000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("threads")
                .short('T')
                .long("threads")
                .alias("processors")
                .help("Number of threads. Defaults to all logical cores")
                .value_name("THREADS")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Random seed for null randomizations")
                .value_name("SEED")
                .default_value("1337")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("quiet-progress")
                .long("quiet-progress")
                .help("Accepted for script compatibility; progress is controlled by RUST_LOG")
                .action(ArgAction::SetTrue),
        )
        .group(
            ArgGroup::new("table")
                .args(["input", "biom"])
                .required(true)
                .multiple(false),
        )
        .get_matches();

    let input_path = m.get_one::<PathBuf>("input");
    let biom_path = m.get_one::<PathBuf>("biom");
    let output_path = m.get_one::<PathBuf>("output").unwrap();
    let permutations = *m.get_one::<usize>("permutations").unwrap();
    let seed = *m.get_one::<u64>("seed").unwrap();
    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(num_cpus::get)
        .max(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("initialize rayon thread pool")?;
    info!("using {} rayon threads", rayon::current_num_threads());

    let t0 = Instant::now();
    let (table_path, input_format) = if let Some(path) = input_path {
        (path, RcBrayInputFormat::Text)
    } else {
        let path = biom_path.expect("BIOM path required");
        (path, RcBrayInputFormat::Biom)
    };
    let samples = read_rc_bray_samples(table_path, input_format)?;
    let total_nnz = samples
        .iter()
        .map(|sample| sample.entries.len())
        .sum::<usize>();
    let total_counts = samples.iter().map(|sample| sample.total).sum::<u64>();
    info!(
        "loaded {} non-empty samples, {} nonzero sample-taxa entries, {} total counts in {} ms",
        samples.len(),
        total_nnz,
        total_counts,
        t0.elapsed().as_millis()
    );

    let opts = RcBrayOptions { permutations, seed };
    let t1 = Instant::now();
    let rc = compute_rc_bray(&samples, &opts)?;
    info!(
        "computed RC Bray with {} permutations in {} ms",
        permutations,
        t1.elapsed().as_millis()
    );

    write_rc_matrix(output_path, &samples, &rc)
        .with_context(|| format!("write output {}", output_path.display()))?;
    info!("wrote {}", output_path.display());

    Ok(())
}
