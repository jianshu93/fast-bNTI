use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use newick::{Newick, one_from_string};
use succparen::{
    bitwise::SparseOneNnd,
    tree::{
        LabelVec,
        balanced_parens::BalancedParensTree,
        traversal::{DepthFirstTraverse, VisitNode},
    },
};

type NwkTree = newick::NewickTree;
pub type SuccBpTree = BalancedParensTree<LabelVec<()>, SparseOneNnd>;

pub enum PhyloTree {
    Simple(SimplePhyloTree),
    Succ(SuccPhyloTree),
}

pub struct SimplePhyloTree {
    pub parent: Vec<usize>,
    pub branch_length: Vec<f32>,
    pub postorder: Vec<usize>,
    pub preorder: Vec<usize>,
    pub leaf_nodes: Vec<usize>,
    tip_to_leaf_ord: HashMap<String, usize>,
}

pub struct SuccPhyloTree {
    pub bp: SuccBpTree,
    pub branch_length: Vec<f32>,
    pub leaf_nodes: Vec<usize>,
    tip_to_leaf_ord: HashMap<String, usize>,
}

impl PhyloTree {
    pub fn from_newick_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).context("read newick")?;
        let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
        let t: NwkTree = one_from_string(&sanitized).context("parse sanitized newick")?;
        Ok(Self::Simple(SimplePhyloTree::from_newick_tree(&t)?))
    }

    pub fn from_newick_path_succ(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).context("read newick")?;
        let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
        let t: NwkTree = one_from_string(&sanitized).context("parse sanitized newick")?;
        Ok(Self::Succ(SuccPhyloTree::from_newick_tree(&t)?))
    }

    pub fn node_count(&self) -> usize {
        match self {
            Self::Simple(tree) => tree.node_count(),
            Self::Succ(tree) => tree.node_count(),
        }
    }

    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Simple(tree) => tree.leaf_count(),
            Self::Succ(tree) => tree.leaf_count(),
        }
    }

    pub fn tip_to_leaf_ord(&self) -> &HashMap<String, usize> {
        match self {
            Self::Simple(tree) => &tree.tip_to_leaf_ord,
            Self::Succ(tree) => &tree.tip_to_leaf_ord,
        }
    }
}

impl SimplePhyloTree {
    fn from_newick_tree(t: &NwkTree) -> Result<Self> {
        let mut max_id = 0usize;
        for v in t.nodes() {
            max_id = max_id.max(v);
        }
        let total = max_id + 1;
        let root = t.root();

        let mut parent = vec![usize::MAX; total];
        let mut branch_length = vec![0.0f32; total];
        let mut children = vec![Vec::<usize>::new(); total];
        let mut visited = vec![false; total];

        visited[root] = true;
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            for &c in t[v].children() {
                if visited[c] {
                    continue;
                }
                visited[c] = true;
                parent[c] = v;
                let len = t[c].branch().copied().unwrap_or(0.0);
                if len < 0.0 {
                    anyhow::bail!("negative branch length at node {c}: {len}");
                }
                branch_length[c] = len;
                children[v].push(c);
                stack.push(c);
            }
        }

        for v in t.nodes() {
            if !visited[v] {
                anyhow::bail!("tree node {v} is not reachable from root {root}");
            }
        }

        let mut preorder = Vec::with_capacity(total);
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            preorder.push(v);
            for &c in children[v].iter().rev() {
                stack.push(c);
            }
        }
        let mut postorder = preorder.clone();
        postorder.reverse();

        let mut leaf_nodes = Vec::new();
        let mut tip_to_leaf_ord = HashMap::new();
        for n in t.nodes() {
            if t[n].is_leaf() {
                let name = t
                    .name(n)
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("L{n}"));
                let leaf_ord = leaf_nodes.len();
                leaf_nodes.push(n);
                tip_to_leaf_ord.insert(name, leaf_ord);
            }
        }

        if leaf_nodes.is_empty() {
            anyhow::bail!("tree has no tips");
        }

        Ok(Self {
            parent,
            branch_length,
            postorder,
            preorder,
            leaf_nodes,
            tip_to_leaf_ord,
        })
    }

    pub fn node_count(&self) -> usize {
        self.parent.len()
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_nodes.len()
    }
}

impl SuccPhyloTree {
    fn from_newick_tree(t: &NwkTree) -> Result<Self> {
        let mut max_id = 0usize;
        for v in t.nodes() {
            max_id = max_id.max(v);
        }

        let mut branch_length = vec![0.0f32];
        let mut original_to_bp = vec![usize::MAX; max_id + 1];
        let trav = SuccTrav::new(t, &mut branch_length, &mut original_to_bp);
        let bp: SuccBpTree =
            BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

        let total = bp.len() + 1;
        branch_length.resize(total, 0.0);
        for (node, &len) in branch_length.iter().enumerate() {
            if len < 0.0 {
                anyhow::bail!("negative branch length at succ node {node}: {len}");
            }
        }

        let mut leaf_nodes = Vec::new();
        let mut tip_to_leaf_ord = HashMap::new();
        for n in t.nodes() {
            if t[n].is_leaf() {
                let mapped = original_to_bp[n];
                if mapped == usize::MAX {
                    anyhow::bail!("tree tip node {n} was not visited by succ traversal");
                }
                let name = t
                    .name(n)
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("L{n}"));
                let leaf_ord = leaf_nodes.len();
                leaf_nodes.push(mapped);
                tip_to_leaf_ord.insert(name, leaf_ord);
            }
        }

        if leaf_nodes.is_empty() {
            anyhow::bail!("tree has no tips");
        }

        Ok(Self {
            bp,
            branch_length,
            leaf_nodes,
            tip_to_leaf_ord,
        })
    }

    pub fn node_count(&self) -> usize {
        self.bp.len() + 1
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_nodes.len()
    }
}

struct SuccTrav<'a> {
    t: &'a NwkTree,
    stack: Vec<(usize, usize, usize)>,
    branch_length: &'a mut Vec<f32>,
    original_to_bp: &'a mut [usize],
    next_bp_id: usize,
}

impl<'a> SuccTrav<'a> {
    fn new(
        t: &'a NwkTree,
        branch_length: &'a mut Vec<f32>,
        original_to_bp: &'a mut [usize],
    ) -> Self {
        Self {
            t,
            stack: vec![(t.root(), 0, 0)],
            branch_length,
            original_to_bp,
            next_bp_id: 1,
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, level, nth) = self.stack.pop()?;
        let bp_id = self.next_bp_id;
        self.next_bp_id += 1;

        let n_children = self.t[id].children().len();
        for (k, &child) in self.t[id].children().iter().enumerate().rev() {
            let nth = n_children - 1 - k;
            self.stack.push((child, level + 1, nth));
        }

        self.original_to_bp[id] = bp_id;
        if self.branch_length.len() <= bp_id {
            self.branch_length.resize(bp_id + 1, 0.0);
        }
        self.branch_length[bp_id] = if id == self.t.root() {
            0.0
        } else {
            self.t[id].branch().copied().unwrap_or(0.0)
        };

        Some(VisitNode::new((), level, nth))
    }
}

// Adapted from DartUniFrac's Newick sanitizer. It removes comments and internal
// support labels while preserving branch lengths and tip labels.
fn sanitize_newick_drop_internal_labels_and_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                i += 1;
                let mut depth = 1;
                while i < bytes.len() && depth > 0 {
                    match bytes[i] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
            }
            b')' => {
                out.push(')');
                i += 1;

                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }

                if i < bytes.len() && bytes[i] == b'\'' {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 2;
                            continue;
                        }
                        if bytes[i] == b'\'' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                } else {
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c.is_ascii_whitespace()
                            || matches!(c, b':' | b',' | b')' | b'(' | b';' | b'[')
                        {
                            break;
                        }
                        i += 1;
                    }
                }
            }
            _ => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    out
}
