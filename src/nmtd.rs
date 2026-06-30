use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::Result;
use rayon::prelude::*;
use succparen::tree::Node as SuccNode;

use crate::{
    table::Sample,
    tree::{PhyloTree, SimplePhyloTree, SuccPhyloTree},
};

const INF: f32 = f32::INFINITY;

pub struct BetaOptions {
    pub block_size: usize,
}

pub fn compute_pair_values(
    tree: &PhyloTree,
    samples: &[Sample],
    permutation: Option<&[usize]>,
    opts: &BetaOptions,
) -> Vec<f64> {
    let n = samples.len();
    let directed = match tree {
        PhyloTree::Simple(tree) => {
            compute_directed_matrix_simple(tree, samples, permutation, opts.block_size.max(1))
        }
        PhyloTree::Succ(tree) => {
            compute_directed_matrix_succ(tree, samples, permutation, opts.block_size.max(1))
        }
    };
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = directed[i * n + j];
            let b = directed[j * n + i];
            out.push(0.5 * (a + b));
        }
    }
    out
}

fn compute_directed_matrix_simple(
    tree: &SimplePhyloTree,
    samples: &[Sample],
    permutation: Option<&[usize]>,
    block_size: usize,
) -> Vec<f64> {
    let n = samples.len();
    let starts = (0..n).step_by(block_size).collect::<Vec<_>>();
    let blocks = starts
        .into_par_iter()
        .map(|start| {
            let width = (n - start).min(block_size);
            let vals = compute_target_block_simple(tree, samples, permutation, start, width);
            (start, width, vals)
        })
        .collect::<Vec<_>>();

    let mut directed = vec![0.0f64; n * n];
    for (start, width, vals) in blocks {
        for i in 0..n {
            let src = i * width;
            let dst = i * n + start;
            directed[dst..dst + width].copy_from_slice(&vals[src..src + width]);
        }
    }
    directed
}

fn compute_target_block_simple(
    tree: &SimplePhyloTree,
    samples: &[Sample],
    permutation: Option<&[usize]>,
    start: usize,
    width: usize,
) -> Vec<f64> {
    let node_count = tree.node_count();
    let mut dist = vec![INF; node_count * width];

    for k in 0..width {
        let target = &samples[start + k];
        for entry in &target.entries {
            let node = mapped_leaf_node_simple(tree, entry.leaf_ord, permutation);
            dist[node * width + k] = 0.0;
        }
    }

    for &node in &tree.postorder {
        let parent = tree.parent[node];
        if parent == usize::MAX {
            continue;
        }
        let edge = tree.branch_length[node];
        let child_base = node * width;
        let parent_base = parent * width;
        for k in 0..width {
            let candidate = dist[child_base + k] + edge;
            if candidate < dist[parent_base + k] {
                dist[parent_base + k] = candidate;
            }
        }
    }

    for &node in &tree.preorder {
        let parent = tree.parent[node];
        if parent == usize::MAX {
            continue;
        }
        let edge = tree.branch_length[node];
        let node_base = node * width;
        let parent_base = parent * width;
        for k in 0..width {
            let candidate = dist[parent_base + k] + edge;
            if candidate < dist[node_base + k] {
                dist[node_base + k] = candidate;
            }
        }
    }

    let mut out = vec![0.0f64; samples.len() * width];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(sample_i, row)| {
            for entry in &samples[sample_i].entries {
                let node = mapped_leaf_node_simple(tree, entry.leaf_ord, permutation);
                let base = node * width;
                let weight = entry.weight as f64;
                for k in 0..width {
                    row[k] += weight * dist[base + k] as f64;
                }
            }
        });

    out
}

fn compute_directed_matrix_succ(
    tree: &SuccPhyloTree,
    samples: &[Sample],
    permutation: Option<&[usize]>,
    block_size: usize,
) -> Vec<f64> {
    let n = samples.len();
    let starts = (0..n).step_by(block_size).collect::<Vec<_>>();
    let blocks = starts
        .into_par_iter()
        .map(|start| {
            let width = (n - start).min(block_size);
            let vals = compute_target_block_succ(tree, samples, permutation, start, width);
            (start, width, vals)
        })
        .collect::<Vec<_>>();

    let mut directed = vec![0.0f64; n * n];
    for (start, width, vals) in blocks {
        for i in 0..n {
            let src = i * width;
            let dst = i * n + start;
            directed[dst..dst + width].copy_from_slice(&vals[src..src + width]);
        }
    }
    directed
}

fn compute_target_block_succ(
    tree: &SuccPhyloTree,
    samples: &[Sample],
    permutation: Option<&[usize]>,
    start: usize,
    width: usize,
) -> Vec<f64> {
    let node_count = tree.node_count();
    let mut dist = vec![INF; node_count * width];

    for k in 0..width {
        let target = &samples[start + k];
        for entry in &target.entries {
            let node = mapped_leaf_node_succ(tree, entry.leaf_ord, permutation);
            dist[node * width + k] = 0.0;
        }
    }

    succ_bottom_up(tree, &mut dist, width);
    succ_top_down(tree, &mut dist, width);

    let mut out = vec![0.0f64; samples.len() * width];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(sample_i, row)| {
            for entry in &samples[sample_i].entries {
                let node = mapped_leaf_node_succ(tree, entry.leaf_ord, permutation);
                let base = node * width;
                let weight = entry.weight as f64;
                for k in 0..width {
                    row[k] += weight * dist[base + k] as f64;
                }
            }
        });

    out
}

fn succ_bottom_up(tree: &SuccPhyloTree, dist: &mut [f32], width: usize) {
    enum Frame<N> {
        Enter { node: N, parent: Option<usize> },
        Exit { node: usize, parent: Option<usize> },
    }

    let mut stack = vec![Frame::Enter {
        node: tree.bp.root(),
        parent: None,
    }];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter { node, parent } => {
                let node_id = node.id() as usize;
                stack.push(Frame::Exit {
                    node: node_id,
                    parent,
                });
                for edge in node.children() {
                    stack.push(Frame::Enter {
                        node: edge.node,
                        parent: Some(node_id),
                    });
                }
            }
            Frame::Exit { node, parent } => {
                let Some(parent) = parent else {
                    continue;
                };
                let edge = tree.branch_length[node];
                let child_base = node * width;
                let parent_base = parent * width;
                for k in 0..width {
                    let candidate = dist[child_base + k] + edge;
                    if candidate < dist[parent_base + k] {
                        dist[parent_base + k] = candidate;
                    }
                }
            }
        }
    }
}

fn succ_top_down(tree: &SuccPhyloTree, dist: &mut [f32], width: usize) {
    let mut stack = vec![tree.bp.root()];
    while let Some(node) = stack.pop() {
        let parent_id = node.id() as usize;
        let parent_base = parent_id * width;
        for edge_node in node.children().map(|edge| edge.node) {
            let child_id = edge_node.id() as usize;
            let edge = tree.branch_length[child_id];
            let child_base = child_id * width;
            for k in 0..width {
                let candidate = dist[parent_base + k] + edge;
                if candidate < dist[child_base + k] {
                    dist[child_base + k] = candidate;
                }
            }
            stack.push(edge_node);
        }
    }
}

fn mapped_leaf_node_simple(
    tree: &SimplePhyloTree,
    leaf_ord: usize,
    permutation: Option<&[usize]>,
) -> usize {
    let mapped = permutation.map(|p| p[leaf_ord]).unwrap_or(leaf_ord);
    tree.leaf_nodes[mapped]
}

fn mapped_leaf_node_succ(
    tree: &SuccPhyloTree,
    leaf_ord: usize,
    permutation: Option<&[usize]>,
) -> usize {
    let mapped = permutation.map(|p| p[leaf_ord]).unwrap_or(leaf_ord);
    tree.leaf_nodes[mapped]
}

pub fn write_long_output(
    path: &Path,
    samples: &[Sample],
    observed: &[f64],
    null_mean: &[f64],
    null_sd: &[f64],
    beta_nti: &[f64],
    permutations: usize,
) -> Result<()> {
    let mut out = BufWriter::with_capacity(16 << 20, File::create(path)?);
    writeln!(
        out,
        "sample1\tsample2\tbeta_mntd\tnull_mean\tnull_sd\tbeta_nti\tpermutations"
    )?;
    let mut idx = 0usize;
    let mut buf = ryu::Buffer::new();
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            out.write_all(samples[i].name.as_bytes())?;
            out.write_all(b"\t")?;
            out.write_all(samples[j].name.as_bytes())?;
            out.write_all(b"\t")?;
            write_f64(&mut out, &mut buf, observed[idx])?;
            out.write_all(b"\t")?;
            write_f64(&mut out, &mut buf, null_mean[idx])?;
            out.write_all(b"\t")?;
            write_f64(&mut out, &mut buf, null_sd[idx])?;
            out.write_all(b"\t")?;
            write_f64(&mut out, &mut buf, beta_nti[idx])?;
            writeln!(out, "\t{permutations}")?;
            idx += 1;
        }
    }
    out.flush()?;
    Ok(())
}

pub fn write_matrix_output(path: &Path, samples: &[Sample], beta_nti: &[f64]) -> Result<()> {
    let n = samples.len();
    let mut mat = vec![0.0f64; n * n];
    let mut idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            mat[i * n + j] = beta_nti[idx];
            mat[j * n + i] = beta_nti[idx];
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
            write_f64(&mut out, &mut buf, mat[i * n + j])?;
        }
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn write_f64<W: Write>(out: &mut W, buf: &mut ryu::Buffer, v: f64) -> Result<()> {
    if v.is_nan() {
        out.write_all(b"nan")?;
    } else if v.is_infinite() && v.is_sign_positive() {
        out.write_all(b"inf")?;
    } else if v.is_infinite() {
        out.write_all(b"-inf")?;
    } else {
        out.write_all(buf.format_finite(v).as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::table::{Sample, SampleEntry};

    #[test]
    fn pair_index_order_is_upper_triangle() {
        let samples = vec![
            Sample {
                name: "a".into(),
                entries: vec![],
            },
            Sample {
                name: "b".into(),
                entries: vec![],
            },
            Sample {
                name: "c".into(),
                entries: vec![],
            },
        ];
        let values = vec![1.0, 2.0, 3.0];
        assert_eq!(samples.len() * (samples.len() - 1) / 2, values.len());
    }

    #[test]
    fn sample_entry_is_copy() {
        let entry = SampleEntry {
            leaf_ord: 1,
            weight: 0.5,
        };
        let copied = entry;
        assert_eq!(copied.leaf_ord, 1);
    }
}
