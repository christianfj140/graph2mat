"""Where the band error of a prediction of a layered structure comes from.

The error dH = H_pred - H_ref (minus the Gamma least-squares gauge shift c*S) is split by
matrix-element class, and each class's first-order effect on the reference bands is computed
along a k path::

    d_eps_n,c(k) = <psi_n(k)| dH_c(k) |psi_n(k)>,   psi S-orthonormal, |eps_n| <= 2 eV.

First order is linear in dH, so the classes add up to the total first-order shift; after
removing the window mean (the rigid shift that aligned band metrics ignore), each class's
share of the aligned error variance is cov(d_eps_c, d_eps_tot) / var(d_eps_tot), summing to
1. The exact aligned band RMSE (diagonalizing H_pred) is reported to check that first order
holds.

Classes: on-site; intralayer 1st/2nd/3rd shell and farther (graphene distances);
interlayer by in-plane offset (stacked < 1 Ang, near < 2 Ang, far); and "beyond":
reference elements in periodic images the prediction does not have (its nsc is smaller),
i.e. -H_ref there. Used to show that the out-of-distribution band error of bilayers
trained on monolayers comes from the interlayer coupling.
"""
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .metrics import HEXAGONAL_PATH, INTERLAYER_DZ_ANG

__all__ = ["CLASSES", "element_classes", "attribute", "counterfactuals"]

CLASSES = (
    "onsite",
    "intra1",
    "intra2",
    "intra3",
    "intra_far",
    "inter_stacked",
    "inter_near",
    "inter_far",
    "beyond",
)
WINDOW_EV = 2.0


def element_classes(h_ref, pred_nsc) -> Tuple[sp.coo_matrix, np.ndarray]:
    """Class index (into ``CLASSES``) of every element of the reference supercell pattern,
    and that pattern."""
    g = h_ref.geometry
    no = g.no
    ref = h_ref.tocsr(0).tocoo()
    rows, cols = ref.row, ref.col
    isc = cols // no
    a_i, a_j = g.o2a(rows), g.o2a(cols % no)
    off = g.lattice.sc_off[isc]
    vec = g.xyz[a_j] + off @ g.cell - g.xyz[a_i]
    dz, rho = np.abs(vec[:, 2]), np.hypot(vec[:, 0], vec[:, 1])
    d = np.linalg.norm(vec, axis=1)
    cls = np.where(
        dz > INTERLAYER_DZ_ANG,
        np.select([rho < 1.0, rho < 2.0], [5, 6], 7),
        np.select([d < 0.1, d < 1.6, d < 2.6, d < 3.0], [0, 1, 2, 3], 4),
    )
    reach = np.asarray(pred_nsc) // 2
    cls[np.any(np.abs(off) > reach, axis=1)] = 8
    return ref, cls


def _unpolarized(*hamiltonians) -> None:
    for h in hamiltonians:
        if not h.spin.is_unpolarized:
            raise NotImplementedError(f"band attribution supports unpolarized matrices only, not {h.spin}")


def _fold(rows, cols, vals, no, sc_off, k) -> np.ndarray:
    phase = np.exp(2j * np.pi * (sc_off[cols // no] @ k))
    return sp.coo_matrix((vals * phase, (rows, cols % no)), shape=(no, no)).toarray()


def _coo(m):
    m = m.tocoo()
    return m.row, m.col, m.data


def attribute(
    predicted: Path, reference: Path, n_per_segment: int = 20, kpath: Sequence = HEXAGONAL_PATH
) -> Dict[str, float]:
    """``c_star_meV``, exact and first-order aligned band RMSE (meV), and per class its
    ``share_<class>`` of the aligned error variance and ``rms_<class>_meV``."""
    import scipy.linalg
    import sisl

    h_ref = sisl.get_sile(str(reference)).read_hamiltonian()
    h_pred = sisl.get_sile(str(predicted)).read_hamiltonian()
    _unpolarized(h_ref, h_pred)
    pred_nsc = h_pred.nsc.copy()
    # Both on the larger supercell: no coupling of either is dropped.
    nsc = np.maximum(h_ref.nsc, h_pred.nsc)
    h_ref.set_nsc(nsc)
    h_pred.set_nsc(nsc)
    g, no = h_ref.geometry, h_ref.no
    sc_off = g.lattice.sc_off
    ref_pat, cls = element_classes(h_ref, pred_nsc)
    delta = (h_pred.tocsr(0) - h_ref.tocsr(0)).tocsr()
    S = h_ref.tocsr(h_ref.S_idx)
    d0, s0 = _fold(*_coo(delta), no, sc_off, np.zeros(3)), _fold(*_coo(S), no, sc_off, np.zeros(3))
    c_star = float(np.real(np.vdot(s0, d0)) / np.real(np.vdot(s0, s0)))
    delta = (delta - c_star * S).tocoo()
    # Prediction-only elements are classified by geometry too (same function on their pattern).
    key = dict(zip(zip(ref_pat.row.tolist(), ref_pat.col.tolist()), cls.tolist()))
    missing = [(r, c) for r, c in zip(delta.row.tolist(), delta.col.tolist()) if (r, c) not in key]
    if missing:
        union = sisl.Hamiltonian.fromsp(
            g, sp.csr_matrix((np.ones(len(missing)), tuple(np.array(missing).T)), shape=delta.shape)
        )
        extra_pat, extra_cls = element_classes(union, pred_nsc)
        key.update(zip(zip(extra_pat.row.tolist(), extra_pat.col.tolist()), extra_cls.tolist()))
    dcls = np.array([key[(r, c)] for r, c in zip(delta.row.tolist(), delta.col.tolist())])

    path = sisl.BandStructure(g, kpath, n_per_segment)
    first, exact = [], []
    for k in path.k:
        hk_ref, sk = h_ref.Hk(k=k, format="array"), h_ref.Sk(k=k, format="array")
        eps, vec = scipy.linalg.eigh(hk_ref, sk)
        sel = np.abs(eps) <= WINDOW_EV
        if not sel.any():
            continue
        v = vec[:, sel]
        per_class = np.zeros((len(CLASSES), sel.sum()))
        for c in range(len(CLASSES)):
            m = dcls == c
            if m.any():
                dk = _fold(delta.row[m], delta.col[m], delta.data[m], no, sc_off, k)
                per_class[c] = np.real(np.einsum("in,ij,jn->n", v.conj(), dk, v))
        first.append(per_class)
        eps_pred = scipy.linalg.eigh(h_pred.Hk(k=k, format="array") - c_star * sk, sk, eigvals_only=True)
        exact.append(eps_pred[sel] - eps[sel])
    first = np.concatenate(first, axis=1)  # classes x (bands * k)
    exact = np.concatenate(exact)
    tot = first.sum(0)
    tot_c, exact_c = tot - tot.mean(), exact - exact.mean()
    first_c = first - first.mean(1, keepdims=True)
    var = float(np.mean(tot_c**2))
    out = {
        "c_star_meV": 1e3 * c_star,
        "band_aligned_exact_meV": 1e3 * float(np.sqrt(np.mean(exact_c**2))),
        "band_aligned_first_order_meV": 1e3 * float(np.sqrt(var)),
    }
    for c, name in enumerate(CLASSES):
        out[f"share_{name}"] = float(np.mean(first_c[c] * tot_c) / var) if var else float("nan")
        out[f"rms_{name}_meV"] = 1e3 * float(np.sqrt(np.mean(first_c[c] ** 2)))
    return out


def counterfactuals(
    predicted: Path, reference: Path, h0: Path, n_per_segment: int = 20, kpath: Sequence = HEXAGONAL_PATH
) -> Dict[str, float]:
    """Exact aligned band RMSE (meV) of: the prediction; the prediction with its interlayer
    elements taken from H0 (no learned correction where training had no pairs); H0 alone; and
    the prediction with exact interlayer elements (the bound of fixing that part)."""
    import scipy.linalg
    import sisl

    h_ref = sisl.get_sile(str(reference)).read_hamiltonian()
    h_pred = sisl.get_sile(str(predicted)).read_hamiltonian()
    h_0 = sisl.get_sile(str(h0)).read_hamiltonian()
    _unpolarized(h_ref, h_pred, h_0)
    pred_nsc = h_pred.nsc.copy()
    nsc = np.maximum.reduce([h_ref.nsc, h_pred.nsc, h_0.nsc])
    for h in (h_ref, h_pred, h_0):
        h.set_nsc(nsc)
    g, no = h_ref.geometry, h_ref.no
    ref, pred, zero = (h.tocsr(0).tocsr() for h in (h_ref, h_pred, h_0))
    pattern = sisl.Hamiltonian.fromsp(g, (abs(ref) + abs(pred) + abs(zero)).tocsr())
    pat, cls = element_classes(pattern, pred_nsc)
    inter = sp.csr_matrix((np.isin(cls, (5, 6, 7)).astype(float), (pat.row, pat.col)), shape=ref.shape)
    S = h_ref.tocsr(h_ref.S_idx)
    variants = {
        "prediction": pred,
        "inter_from_H0": pred + inter.multiply(zero - pred),
        "H0_only": zero,
        "inter_exact": pred + inter.multiply(ref - pred),
    }
    path = sisl.BandStructure(g, kpath, n_per_segment)
    sc_off = g.lattice.sc_off
    s0 = _fold(*_coo(S), no, sc_off, np.zeros(3))
    out = {}
    for name, h in variants.items():
        d0 = _fold(*_coo(h - ref), no, sc_off, np.zeros(3))
        c_star = float(np.real(np.vdot(s0, d0)) / np.real(np.vdot(s0, s0)))
        errs = []
        for k in path.k:
            sk = _fold(*_coo(S), no, sc_off, k)
            eps = scipy.linalg.eigh(_fold(*_coo(ref), no, sc_off, k), sk, eigvals_only=True)
            sel = np.abs(eps) <= WINDOW_EV
            if sel.any():
                hk = _fold(*_coo(h), no, sc_off, k) - c_star * sk
                errs.append(scipy.linalg.eigh(hk, sk, eigvals_only=True)[sel] - eps[sel])
        e = np.concatenate(errs)
        out[f"cf_{name}_meV"] = 1e3 * float(np.sqrt(np.mean((e - e.mean()) ** 2)))
    return out
