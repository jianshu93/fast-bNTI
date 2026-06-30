use std::{
    cell::RefCell,
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

thread_local! {
    static DIST_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub struct BetaOptions {
    pub block_size: usize,
}

#[derive(Copy, Clone, Debug)]
enum SimdBackend {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
}

impl SimdBackend {
    fn detect_dot(width: usize) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if width >= 8 && std::arch::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if width >= 4 && std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if width >= 2 {
                return Self::Neon;
            }
        }

        #[allow(unreachable_code)]
        Self::Scalar
    }

    fn detect_tree(width: usize) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if width >= 16 && std::arch::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if width >= 8 && std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if width >= 4 {
                return Self::Neon;
            }
        }

        #[allow(unreachable_code)]
        Self::Scalar
    }

    #[inline(always)]
    fn accumulate(self, row: &mut [f64], dist: &[f32], weight: f64) {
        debug_assert_eq!(row.len(), dist.len());
        match self {
            Self::Scalar => accumulate_weighted_dist_scalar(row, dist, weight),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => accumulate_weighted_dist_neon(row, dist, weight),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { accumulate_weighted_dist_avx2(row, dist, weight) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512 => unsafe { accumulate_weighted_dist_avx512(row, dist, weight) },
        }
    }

    #[inline(always)]
    fn relax_min(self, dst: &mut [f32], src: &[f32], edge: f32) {
        debug_assert_eq!(dst.len(), src.len());
        match self {
            Self::Scalar => relax_min_scalar(dst, src, edge),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => relax_min_neon(dst, src, edge),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { relax_min_avx2(dst, src, edge) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512 => unsafe { relax_min_avx512(dst, src, edge) },
        }
    }
}

#[inline(always)]
fn accumulate_weighted_dist_scalar(row: &mut [f64], dist: &[f32], weight: f64) {
    for (acc, &d) in row.iter_mut().zip(dist.iter()) {
        *acc += weight * d as f64;
    }
}

#[inline(always)]
fn relax_min_block(
    backend: SimdBackend,
    dist: &mut [f32],
    dst_base: usize,
    src_base: usize,
    width: usize,
    edge: f32,
) {
    debug_assert_ne!(dst_base, src_base);
    debug_assert!(dst_base + width <= dist.len());
    debug_assert!(src_base + width <= dist.len());

    if dst_base < src_base {
        let (left, right) = dist.split_at_mut(src_base);
        let dst = &mut left[dst_base..dst_base + width];
        let src = &right[..width];
        backend.relax_min(dst, src, edge);
    } else {
        let (left, right) = dist.split_at_mut(dst_base);
        let src = &left[src_base..src_base + width];
        let dst = &mut right[..width];
        backend.relax_min(dst, src, edge);
    }
}

#[inline(always)]
fn relax_min_scalar(dst: &mut [f32], src: &[f32], edge: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let candidate = s + edge;
        if candidate < *d {
            *d = candidate;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn relax_min_neon(dst: &mut [f32], src: &[f32], edge: f32) {
    use core::arch::aarch64::*;

    let mut k = 0usize;
    let width = dst.len();
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        let edge_v = vdupq_n_f32(edge);
        while k + 4 <= width {
            let src_v = vld1q_f32(src_ptr.add(k));
            let dst_v = vld1q_f32(dst_ptr.add(k));
            let candidate = vaddq_f32(src_v, edge_v);
            vst1q_f32(dst_ptr.add(k), vminq_f32(dst_v, candidate));
            k += 4;
        }
    }
    if k < width {
        relax_min_scalar(&mut dst[k..], &src[k..], edge);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relax_min_avx2(dst: &mut [f32], src: &[f32], edge: f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut k = 0usize;
    let width = dst.len();
    let edge_v = _mm256_set1_ps(edge);
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        while k + 8 <= width {
            let src_v = _mm256_loadu_ps(src_ptr.add(k));
            let dst_v = _mm256_loadu_ps(dst_ptr.add(k));
            let candidate = _mm256_add_ps(src_v, edge_v);
            _mm256_storeu_ps(dst_ptr.add(k), _mm256_min_ps(dst_v, candidate));
            k += 8;
        }
    }
    if k < width {
        relax_min_scalar(&mut dst[k..], &src[k..], edge);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn relax_min_avx512(dst: &mut [f32], src: &[f32], edge: f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut k = 0usize;
    let width = dst.len();
    let edge_v = _mm512_set1_ps(edge);
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        while k + 16 <= width {
            let src_v = _mm512_loadu_ps(src_ptr.add(k));
            let dst_v = _mm512_loadu_ps(dst_ptr.add(k));
            let candidate = _mm512_add_ps(src_v, edge_v);
            _mm512_storeu_ps(dst_ptr.add(k), _mm512_min_ps(dst_v, candidate));
            k += 16;
        }
    }
    if k < width {
        relax_min_scalar(&mut dst[k..], &src[k..], edge);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn accumulate_weighted_dist_neon(row: &mut [f64], dist: &[f32], weight: f64) {
    use core::arch::aarch64::*;

    let mut k = 0usize;
    let width = row.len();
    unsafe {
        let row_ptr = row.as_mut_ptr();
        let dist_ptr = dist.as_ptr();
        while k + 2 <= width {
            let d32 = vld1_f32(dist_ptr.add(k));
            let d64 = vcvt_f64_f32(d32);
            let acc = vld1q_f64(row_ptr.add(k));
            let next = vaddq_f64(acc, vmulq_n_f64(d64, weight));
            vst1q_f64(row_ptr.add(k), next);
            k += 2;
        }
    }
    if k < width {
        accumulate_weighted_dist_scalar(&mut row[k..], &dist[k..], weight);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_weighted_dist_avx2(row: &mut [f64], dist: &[f32], weight: f64) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut k = 0usize;
    let width = row.len();
    let w = _mm256_set1_pd(weight);
    unsafe {
        let row_ptr = row.as_mut_ptr();
        let dist_ptr = dist.as_ptr();
        while k + 4 <= width {
            let d32 = _mm_loadu_ps(dist_ptr.add(k));
            let d64 = _mm256_cvtps_pd(d32);
            let acc = _mm256_loadu_pd(row_ptr.add(k));
            let next = _mm256_add_pd(acc, _mm256_mul_pd(w, d64));
            _mm256_storeu_pd(row_ptr.add(k), next);
            k += 4;
        }
    }
    if k < width {
        accumulate_weighted_dist_scalar(&mut row[k..], &dist[k..], weight);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_weighted_dist_avx512(row: &mut [f64], dist: &[f32], weight: f64) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut k = 0usize;
    let width = row.len();
    let w = _mm512_set1_pd(weight);
    unsafe {
        let row_ptr = row.as_mut_ptr();
        let dist_ptr = dist.as_ptr();
        while k + 8 <= width {
            let d32 = _mm256_loadu_ps(dist_ptr.add(k));
            let d64 = _mm512_cvtps_pd(d32);
            let acc = _mm512_loadu_pd(row_ptr.add(k));
            let next = _mm512_add_pd(acc, _mm512_mul_pd(w, d64));
            _mm512_storeu_pd(row_ptr.add(k), next);
            k += 8;
        }
    }
    if k < width {
        accumulate_weighted_dist_scalar(&mut row[k..], &dist[k..], weight);
    }
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
    let dot_backend = SimdBackend::detect_dot(width);
    let tree_backend = SimdBackend::detect_tree(width);
    let mut dist = take_dist_scratch(node_count * width);

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
        relax_min_block(
            tree_backend,
            &mut dist,
            parent_base,
            child_base,
            width,
            edge,
        );
    }

    for &node in &tree.preorder {
        let parent = tree.parent[node];
        if parent == usize::MAX {
            continue;
        }
        let edge = tree.branch_length[node];
        let node_base = node * width;
        let parent_base = parent * width;
        relax_min_block(tree_backend, &mut dist, node_base, parent_base, width, edge);
    }

    let mut out = vec![0.0f64; samples.len() * width];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(sample_i, row)| {
            for entry in &samples[sample_i].entries {
                let node = mapped_leaf_node_simple(tree, entry.leaf_ord, permutation);
                let base = node * width;
                dot_backend.accumulate(row, &dist[base..base + width], entry.weight as f64);
            }
        });

    put_dist_scratch(dist);
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
    let dot_backend = SimdBackend::detect_dot(width);
    let tree_backend = SimdBackend::detect_tree(width);
    let mut dist = take_dist_scratch(node_count * width);

    for k in 0..width {
        let target = &samples[start + k];
        for entry in &target.entries {
            let node = mapped_leaf_node_succ(tree, entry.leaf_ord, permutation);
            dist[node * width + k] = 0.0;
        }
    }

    succ_bottom_up(tree, &mut dist, width, tree_backend);
    succ_top_down(tree, &mut dist, width, tree_backend);

    let mut out = vec![0.0f64; samples.len() * width];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(sample_i, row)| {
            for entry in &samples[sample_i].entries {
                let node = mapped_leaf_node_succ(tree, entry.leaf_ord, permutation);
                let base = node * width;
                dot_backend.accumulate(row, &dist[base..base + width], entry.weight as f64);
            }
        });

    put_dist_scratch(dist);
    out
}

fn take_dist_scratch(len: usize) -> Vec<f32> {
    let mut dist = DIST_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        std::mem::take(&mut *scratch)
    });
    if dist.len() != len {
        dist.resize(len, INF);
    }
    dist.fill(INF);
    dist
}

fn put_dist_scratch(mut dist: Vec<f32>) {
    dist.clear();
    DIST_SCRATCH.with(|scratch| {
        let mut slot = scratch.borrow_mut();
        if dist.capacity() > slot.capacity() {
            *slot = dist;
        }
    });
}

fn succ_bottom_up(tree: &SuccPhyloTree, dist: &mut [f32], width: usize, backend: SimdBackend) {
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
                relax_min_block(backend, dist, parent_base, child_base, width, edge);
            }
        }
    }
}

fn succ_top_down(tree: &SuccPhyloTree, dist: &mut [f32], width: usize, backend: SimdBackend) {
    let mut stack = vec![tree.bp.root()];
    while let Some(node) = stack.pop() {
        let parent_id = node.id() as usize;
        let parent_base = parent_id * width;
        for edge_node in node.children().map(|edge| edge.node) {
            let child_id = edge_node.id() as usize;
            let edge = tree.branch_length[child_id];
            let child_base = child_id * width;
            relax_min_block(backend, dist, child_base, parent_base, width, edge);
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
    use super::{SimdBackend, accumulate_weighted_dist_scalar, relax_min_scalar};
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

    #[test]
    fn simd_accumulator_matches_scalar() {
        let dist = vec![0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0];
        let mut scalar = vec![0.5f64; dist.len()];
        let mut selected = scalar.clone();

        accumulate_weighted_dist_scalar(&mut scalar, &dist, 0.125);
        SimdBackend::detect_dot(dist.len()).accumulate(&mut selected, &dist, 0.125);

        for (a, b) in scalar.iter().zip(selected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn simd_relax_min_matches_scalar() {
        let src = vec![0.0, 4.0, 1.5, f32::INFINITY, 3.25, 9.0, 0.125, 2.0, 5.5];
        let mut scalar = vec![10.0, 1.0, 8.0, 2.0, 7.0, 8.5, 0.25, 8.0, 4.0];
        let mut selected = scalar.clone();

        relax_min_scalar(&mut scalar, &src, 0.5);
        SimdBackend::detect_tree(src.len()).relax_min(&mut selected, &src, 0.5);

        assert_eq!(scalar, selected);
    }
}
