"""
Two-site greedy cross interpolation in the TT format (NumPy port).

Python translation of TT-Toolbox/cross/greedy2_cross.m
    D. Savostyanov, "Quasioptimality of maximum-volume cross interpolation
    of tensors", Linear Algebra Appl. 458 (2014) 217-244.
Original MATLAB: I. Oseledets, S. Dolgov, D. Savostyanov.

---------------------------------------------------
* All ``reshape`` calls use Fortran (column-major) order to reproduce the
  MATLAB memory layout the algorithm depends on.
* The returned tensor ``y`` is a list of ``d`` TT-cores (3-D ``float`` arrays
  of shape ``(r_k, n_k, r_{k+1})``) instead of a ``tt_tensor`` object -- this
  is the direct analogue of the MATLAB ``cell2core`` output.
* The optional ``aux`` argument is a list of auxiliary TT-tensors, each given
  in the same "list of cores" format (the analogue of ``core2cell``).

"""

import numpy as np


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _rs(a, shape):
    """MATLAB-style (column-major) reshape."""
    return np.reshape(a, shape, order="F")


def indexmerge(*idxs):
    """Merge two or three index sets in the little-endian (first-fastest) way.

    Each argument is a 2-D integer array of shape ``(rows, n_modes)`` (a 1-D
    array is treated as a single column).  An empty set (0 rows) contributes
    no columns but counts as a single "empty" row, exactly like the MATLAB
    ``max(size(x,1),1)`` trick.  The row ordering of the result is the
    Fortran flattening of shape ``(rows_1, rows_2[, rows_3])``.
    """
    norm = []
    for a in idxs:
        a = np.asarray(a, dtype=np.int64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        norm.append(a)
    sz = [max(a.shape[0], 1) for a in norm]
    total = int(np.prod(sz))
    out_cols = []
    for k, a in enumerate(norm):
        if a.shape[1] == 0 or a.shape[0] == 0:
            continue                       # empty set -> no columns
        inner = int(np.prod(sz[:k]))       # consecutive repeats of each row
        outer = int(np.prod(sz[k + 1:]))   # repeats of the whole block
        expanded = np.repeat(a, inner, axis=0)
        expanded = np.tile(expanded, (outer, 1))
        out_cols.append(expanded)
    if not out_cols:
        return np.zeros((total, 0), dtype=np.int64)
    return np.hstack(out_cols)


def tt_ind2sub(siz, idx):
    """Vectorised linear-index -> multi-index (0-based, Fortran order)."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    siz = np.asarray(siz, dtype=np.int64)
    sub = np.empty((idx.size, siz.size), dtype=np.int64)
    rem = idx.copy()
    for k in range(siz.size):
        sub[:, k] = rem % siz[k]
        rem = rem // siz[k]
    return sub


def _lu_row_pivots(a):
    """Row permutation of partial-pivoted LU (analogue of ``lu(a,'vector')``)."""
    a = np.array(a, dtype=float, copy=True)
    m, r = a.shape
    p = np.arange(m)
    for k in range(min(m, r)):
        piv = k + int(np.argmax(np.abs(a[k:, k])))
        if piv != k:
            a[[k, piv], :] = a[[piv, k], :]
            p[[k, piv]] = p[[piv, k]]
        if a[k, k] != 0.0:
            a[k + 1:, k] /= a[k, k]
            a[k + 1:, k + 1:] -= np.outer(a[k + 1:, k], a[k, k + 1:])
    return p


def maxvol2(a, ind=None, do_qr=False, eps=5e-2, niters=100):
    """Maximal-volume row subset of a tall matrix ``a`` (n x r, n >= r).

    Port of TT-Toolbox/core/maxvol2.m.  Returns the sorted row indices.
    (Like the MATLAB loop, iterations run until the volume can no longer be
    increased; ``niters`` is only a safety cap here.)
    """
    a = np.asarray(a, dtype=float)
    n, r = a.shape
    if n <= r:
        return np.arange(n)
    if do_qr:
        a, _ = np.linalg.qr(a)
    if ind is None:
        ind = _lu_row_pivots(a)[:r].copy()
    ind = np.asarray(ind, dtype=np.int64).copy()

    sbm = a[ind, :]
    b = np.linalg.solve(sbm.T, a.T).T          # a * inv(sbm)

    cap = max(niters, 4 * n)
    for _ in range(cap):
        big = int(np.argmax(np.abs(b.reshape(-1, order="F"))))
        i0, j0 = np.unravel_index(big, b.shape, order="F")
        if np.abs(b[i0, j0]) <= 1.0 + eps:
            break
        b = b + np.outer(b[:, j0], (b[ind[j0], :] - b[i0, :])) / b[i0, j0]
        ind[j0] = i0
    return np.sort(ind)


def autovecfun(fun, J, vec):
    """Evaluate ``fun`` at the rows of ``J`` (0-based multi-indices)."""
    J = np.atleast_2d(np.asarray(J, dtype=np.int64))
    if vec:
        return np.asarray(fun(J), dtype=float).ravel()
    return np.array([float(fun(J[j, :])) for j in range(J.shape[0])], dtype=float)


def formtensor(y, mid_inv, d, ry, n):
    """Fold the factorised inverse interpolation matrices into the fibres."""
    out = []
    for i in range(d):
        yi = _rs(y[i], (ry[i], n[i] * ry[i + 1]))
        yi = mid_inv[i][1] @ yi                       # inv(L) on the left rank
        yi = _rs(yi, (ry[i] * n[i], ry[i + 1]))
        yi = yi @ mid_inv[i + 1][0]                   # inv(U) on the right rank
        out.append(_rs(yi, (ry[i], n[i], ry[i + 1])))
    return out


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def greedy2_cross(n, fun, tol, nswp=20, tol_exit=None, verb=1, vec=False,
                  aux=None, auxfun=None, locsearch="lot", y0=None, xtru=None):
    """Two-site greedy cross interpolation of a black-box tensor in TT format.

    Parameters
    ----------
    n : sequence of int
        Mode sizes, length ``d``.
    fun : callable
        The sought tensor entry as a function of a 0-based multi-index.
        If ``vec`` is False, ``fun(ind)`` takes a length-``d`` index and
        returns one value.  If ``vec`` is True, ``fun(J)`` takes an
        ``(M, d)`` array and returns ``M`` values.
    tol : float
        Target relative accuracy.
    nswp : int, optional
        Maximal number of forward half-sweeps (default 20).
    tol_exit : float, optional
        Stopping threshold on the per-sweep max relative increment
        (default ``tol``).
    verb : int, optional
        0 silent, 1 sweep info, 2 block info.
    vec : bool, optional
        Whether ``fun`` is vectorised (see above).
    aux, auxfun : optional
        ``aux`` is a list of auxiliary TT-tensors (each a list of cores);
        the interpolated tensor is ``fun(ind) + auxfun(aux(ind))`` where
        ``auxfun`` maps an ``(M, len(aux))`` array to ``M`` values.
    locsearch : {'lot', 'als'}, optional
        Pivot search inside a superblock: random lottery ('lot', default,
        fewer evaluations) or 2-D maxvol ALS ('als').
    y0 : sequence of int, optional
        Initial (0-based) left pivots, length ``d-1``.
    xtru : list of cores, optional
        Reference TT tensor; if given (and ``verb>0``) the true error is
        printed each sweep.

    Returns
    -------
    y : list of ndarray
        The TT-cores of the interpolant (shapes ``(r_k, n_k, r_{k+1})``).
    Jyl, Jyr, ilocl, ilocr : lists
        Pivot index sets (left/right global, left/right local).
    evalcnt : int
        Total number of ``fun`` evaluations.
    """
    if tol_exit is None:
        tol_exit = tol
    n = np.asarray(n, dtype=np.int64)
    d = int(n.size)
    ry = np.ones(d + 1, dtype=np.int64)
    y = [None] * d

    # ----- optional auxiliary tensors -------------------------------------
    use_aux = aux is not None
    if use_aux:
        Raux = len(aux)
        raux = np.ones((d + 1, Raux), dtype=np.int64)
        aux_cores = [[None] * Raux for _ in range(d)]
        phiauxl = [[None] * Raux for _ in range(d + 1)]
        phiauxr = [[None] * Raux for _ in range(d + 1)]
        for j in range(Raux):
            ttj = aux[j]
            raux[0, j] = ttj[0].shape[0]
            for i in range(d):
                aux_cores[i][j] = np.asarray(ttj[i], dtype=float)
                raux[i + 1, j] = ttj[i].shape[2]
            phiauxl[0][j] = np.array([[1.0]])
            phiauxl[d][j] = np.array([[1.0]])
            phiauxr[0][j] = np.array([[1.0]])
            phiauxr[d][j] = np.array([[1.0]])

    # Factorised inverse interpolation matrices: mid_inv[k] = [inv(U), inv(L)]
    mid_inv = [[np.array([[1.0]]), np.array([[1.0]])] for _ in range(d + 1)]

    Jyr = [np.zeros((0, 0), dtype=np.int64) for _ in range(d + 1)]
    Jyl = [np.zeros((0, 0), dtype=np.int64) for _ in range(d + 1)]
    ilocr = [np.zeros(0, dtype=np.int64) for _ in range(d + 1)]
    ilocl = [np.zeros(0, dtype=np.int64) for _ in range(d + 1)]

    evalcnt = 0

    def aux_block(i):
        """Full auxiliary contribution over superblock i, shape (r*n*r, Raux)."""
        craux = np.zeros((ry[i] * n[i] * ry[i + 1], Raux))
        for j in range(Raux):
            c1 = _rs(aux_cores[i][j], (raux[i, j], n[i] * raux[i + 1, j]))
            c1 = phiauxl[i][j] @ c1
            c1 = _rs(c1, (ry[i] * n[i], raux[i + 1, j]))
            c1 = c1 @ phiauxr[i + 1][j]
            craux[:, j] = _rs(c1, (ry[i] * n[i] * ry[i + 1], 1)).ravel()
        return craux

    # ----- seed with random left indices ----------------------------------
    for i in range(d - 1):
        if y0 is None:
            v = int(round(np.random.rand() * ry[i] * n[i]))
            if v == 0:
                v = 1
            pos = v - 1
        else:
            pos = int(y0[i])
        ilocl[i + 1] = np.array([pos], dtype=np.int64)
        merged = indexmerge(Jyl[i], np.arange(n[i]))
        Jyl[i + 1] = merged[ilocl[i + 1], :]
        if use_aux:
            for j in range(Raux):
                c1 = _rs(aux_cores[i][j], (raux[i, j], n[i] * raux[i + 1, j]))
                c1 = phiauxl[i][j] @ c1
                c1 = _rs(c1, (ry[i] * n[i], raux[i + 1, j]))
                phiauxl[i + 1][j] = c1[ilocl[i + 1], :]

    # ----- one backward sweep: fix the right indices ----------------------
    for i in range(d - 1, -1, -1):
        J = indexmerge(Jyl[i], np.arange(n[i]), Jyr[i + 1])
        evalcnt += J.shape[0]
        cry1 = autovecfun(fun, J, vec)
        if use_aux:
            cry1 = cry1 + auxfun(aux_block(i))
        if i > 0:
            cm = _rs(cry1, (ry[i], n[i] * ry[i + 1]))
            ilocr[i] = np.argmax(np.abs(cm.T), axis=0).astype(np.int64)
            mid_inv[i][0] = 1.0 / cm[:, ilocr[i]]
            mid_inv[i][1] = np.array([[1.0]])
            Jr = indexmerge(np.arange(n[i]), Jyr[i + 1])
            Jyr[i] = Jr[ilocr[i], :]
            if use_aux:
                for j in range(Raux):
                    p = _rs(aux_cores[i][j], (raux[i, j] * n[i], raux[i + 1, j]))
                    p = p @ phiauxr[i + 1][j]
                    p = _rs(p, (raux[i, j], n[i] * ry[i + 1]))
                    phiauxr[i][j] = p[:, ilocr[i]]
        y[i] = _rs(cry1, (ry[i], n[i], ry[i + 1]))

    # ----- one forward sweep: fix the left indices ------------------------
    for i in range(d):
        J = indexmerge(Jyl[i], np.arange(n[i]), Jyr[i + 1])
        evalcnt += J.shape[0]
        cry1 = autovecfun(fun, J, vec)
        if use_aux:
            cry1 = cry1 + auxfun(aux_block(i))
        if i < d - 1:
            cm = _rs(cry1, (ry[i] * n[i], ry[i + 1]))
            ilocl[i + 1] = np.argmax(np.abs(cm), axis=0).astype(np.int64)
            mid_inv[i + 1][0] = np.array([[1.0 / cm[ilocl[i + 1][0], 0]]])
            mid_inv[i + 1][1] = np.array([[1.0]])
            merged = indexmerge(Jyl[i], np.arange(n[i]))
            Jyl[i + 1] = merged[ilocl[i + 1], :]
            if use_aux:
                for j in range(Raux):
                    p = _rs(aux_cores[i][j], (raux[i, j], n[i] * raux[i + 1, j]))
                    p = phiauxl[i][j] @ p
                    p = _rs(p, (ry[i] * n[i], raux[i + 1, j]))
                    phiauxl[i + 1][j] = p[ilocl[i + 1], :]
        y[i] = _rs(cry1, (ry[i], n[i], ry[i + 1]))

    # ----- main greedy forward half-sweeps --------------------------------
    maxy = 0.0
    max_dx = 0.0
    swp = 1
    dir_ = 1
    i = 0
    while swp <= nswp:
        # candidate index sets with the current crosses removed
        cind1 = np.arange(ry[i] * n[i])
        cind2 = np.arange(n[i + 1] * ry[i + 2])
        cind1 = np.delete(cind1, ilocl[i + 1])
        cind2 = np.delete(cind2, ilocr[i + 1])

        if locsearch == "als":
            testsz = min(cind1.size, cind2.size)
        else:
            N = cind1.size * cind2.size
            m = min(cind1.size, cind2.size)
            t = np.round(np.random.rand(m) * N).astype(np.int64)
            t[t > N] = N
            t[t < 1] = 1
            t = np.unique(t)
            tind0 = t - 1                              # 0-based linear indices
            testsz = tind0.size

        if use_aux:
            craux1 = [None] * Raux
            craux2 = [None] * Raux
            for j in range(Raux):
                c = _rs(aux_cores[i][j], (raux[i, j], n[i] * raux[i + 1, j]))
                c = phiauxl[i][j] @ c
                craux1[j] = _rs(c, (ry[i] * n[i], raux[i + 1, j]))
                c = _rs(aux_cores[i + 1][j],
                        (raux[i + 1, j] * n[i + 1], raux[i + 2, j]))
                c = c @ phiauxr[i + 2][j]
                craux2[j] = _rs(c, (raux[i + 1, j], n[i + 1] * ry[i + 2]))

        if testsz > 0:
            cry1 = _rs(y[i], (ry[i] * n[i], ry[i + 1]))
            cry2 = _rs(y[i + 1], (ry[i + 1], n[i + 1] * ry[i + 2]))

            if locsearch == "als":
                # ---- two-dimensional maxvol ALS ----
                ys1 = (cry1 @ mid_inv[i + 1][0])[cind1, :]
                ys2 = (mid_inv[i + 1][1] @ cry2)[:, cind2]
                J1 = indexmerge(Jyl[i], np.arange(n[i]))
                J1c = J1[cind1, :]
                J2 = indexmerge(np.arange(n[i + 1]), Jyr[i + 2])
                J2c = J2[cind2, :]
                rz = 2                     # target enrichment rank
                cre2 = np.random.randn(cind2.size, rz)
                cre2, _ = np.linalg.qr(cre2)
                indr = maxvol2(cre2)
                rzr = indr.size            # QR/maxvol may return < rz near the edge
                Jr = J2c[indr, :]
                Ye2 = ys2[:, indr]

                J = indexmerge(J1c, Jr)
                evalcnt += J.shape[0]
                cre1 = autovecfun(fun, J, vec)
                if use_aux:
                    ca = np.zeros((cind1.size * indr.size, Raux))
                    for j in range(Raux):
                        M = craux1[j][cind1, :] @ craux2[j][:, cind2[indr]]
                        ca[:, j] = _rs(M, (-1, 1)).ravel()
                    cre1 = cre1 + auxfun(ca)
                maxy = max(maxy, float(np.max(np.abs(cre1))))
                cre1 = _rs(cre1, (cind1.size, rzr)) - ys1 @ Ye2
                zmax1 = float(np.max(np.abs(cre1)))
                sub = tt_ind2sub([cind1.size, rzr],
                                 int(np.argmax(np.abs(cre1).reshape(-1, order="F"))))
                imax1_local = int(sub[0, 0])
                q1, _ = np.linalg.qr(cre1)
                indl = maxvol2(q1)
                rzl = indl.size
                Jl = J1c[indl, :]
                Ye1 = ys1[indl, :]

                J = indexmerge(Jl, J2c)
                evalcnt += J.shape[0]
                cre2 = autovecfun(fun, J, vec)
                if use_aux:
                    ca = np.zeros((indl.size * cind2.size, Raux))
                    for j in range(Raux):
                        M = craux1[j][cind1[indl], :] @ craux2[j][:, cind2]
                        ca[:, j] = _rs(M, (-1, 1)).ravel()
                    cre2 = cre2 + auxfun(ca)
                maxy = max(maxy, float(np.max(np.abs(cre2))))
                cre2 = _rs(cre2, (rzl, cind2.size)) - Ye1 @ ys2
                zmax2 = float(np.max(np.abs(cre2)))
                sub = tt_ind2sub([rzl, cind2.size],
                                 int(np.argmax(np.abs(cre2).reshape(-1, order="F"))))
                imax2_local = int(sub[0, 1])
                imax1 = int(cind1[imax1_local])
                imax2 = int(cind2[imax2_local])
                emax = max(zmax1, zmax2)
                J1full = J1
                J2full = J2
            else:
                # ---- random lottery ----
                sub = tt_ind2sub([cind1.size, cind2.size], tind0)
                pairs = np.column_stack([cind1[sub[:, 0]], cind2[sub[:, 1]]])
                J1 = indexmerge(Jyl[i], np.arange(n[i]))
                J2 = indexmerge(np.arange(n[i + 1]), Jyr[i + 2])
                J1c = J1[pairs[:, 0], :]
                J2c = J2[pairs[:, 1], :]
                J = np.hstack([J1c, J2c])
                evalcnt += testsz
                crt = autovecfun(fun, J, vec)
                if use_aux:
                    ca = np.zeros((testsz, Raux))
                    for j in range(Raux):
                        A = craux1[j][pairs[:, 0], :]
                        B = craux2[j][:, pairs[:, 1]].T
                        ca[:, j] = np.sum(A * B, axis=1)
                    crt = crt + auxfun(ca)
                maxy = max(maxy, float(np.max(np.abs(crt))))

                cre1 = cry1 @ mid_inv[i + 1][0]
                cre2 = mid_inv[i + 1][1] @ cry2
                cry = np.sum(cre1[pairs[:, 0], :] * cre2[:, pairs[:, 1]].T, axis=1)
                cre = crt - cry
                imax2_local = int(np.argmax(np.abs(cre)))

                # maximise the error along the chosen column, over all of cind1
                J1c_full = J1[cind1, :]
                J = indexmerge(J1c_full, J2c[[imax2_local], :])
                evalcnt += J.shape[0]
                crt = autovecfun(fun, J, vec)
                if use_aux:
                    ca = np.zeros((cind1.size, Raux))
                    col = pairs[imax2_local, 1]
                    for j in range(Raux):
                        ca[:, j] = craux1[j][cind1, :] @ craux2[j][:, col]
                    crt = crt + auxfun(ca)
                maxy = max(maxy, float(np.max(np.abs(crt))))
                cry = cre1[cind1, :] @ cre2[:, pairs[imax2_local, 1]]
                cre = crt - cry
                imax1_local = int(np.argmax(np.abs(cre)))
                emax = float(np.abs(cre)[imax1_local])
                imax1 = int(cind1[imax1_local])
                imax2 = int(pairs[imax2_local, 1])
                J1full = J1
                J2full = J2

            dx = emax / maxy
            max_dx = max(max_dx, dx)
            if verb > 1:
                print("=greedy_cross= i=%d, swp=%d, testsz=%d, emax=%.3e, dx=%.3e"
                      % (i, swp, testsz, emax, dx))

            if dx > tol:
                # evaluate the two new fibres
                J1m = J1full[[imax1], :]
                J2m = J2full[[imax2], :]
                Jl = indexmerge(J1full, J2m)
                Jr = indexmerge(J1m, J2full)

                evalcnt += Jl.shape[0]
                cre1 = autovecfun(fun, Jl, vec)
                if use_aux:
                    ca = np.zeros((ry[i] * n[i], Raux))
                    for j in range(Raux):
                        ca[:, j] = craux1[j] @ craux2[j][:, imax2]
                    cre1 = cre1 + auxfun(ca)

                evalcnt += Jr.shape[0]
                cre2 = autovecfun(fun, Jr, vec)
                if use_aux:
                    ca = np.zeros((n[i + 1] * ry[i + 2], Raux))
                    for j in range(Raux):
                        ca[:, j] = craux1[j][imax1, :] @ craux2[j]
                    cre2 = cre2 + auxfun(ca)

                cre1 = _rs(cre1, (ry[i] * n[i], 1))
                cre2 = _rs(cre2, (1, n[i + 1] * ry[i + 2]))
                cry1 = _rs(y[i], (ry[i] * n[i], ry[i + 1]))
                cry2 = _rs(y[i + 1], (ry[i + 1], n[i + 1] * ry[i + 2]))

                y[i] = np.hstack([cry1, cre1])
                y[i + 1] = np.vstack([cry2, cre2])
                ry[i + 1] += 1

                # ilocl[i+2] indexes ry(i+1)*n(i+1); ry(i+1) just grew -> shift
                if i < d - 2:
                    old_r = ry[i + 1] - 1
                    ilocl[i + 2] = ilocl[i + 2] + ilocl[i + 2] // old_r

                Jyl[i + 1] = np.vstack([Jyl[i + 1], J1m])
                Jyr[i + 1] = np.vstack([Jyr[i + 1], J2m])
                ilocl[i + 1] = np.concatenate([ilocl[i + 1], [imax1]])
                ilocr[i + 1] = np.concatenate([ilocr[i + 1], [imax2]])

                # rank-1 analytic LU update of the inverse interpolation matrix
                r_new = int(ry[i + 1])
                r_old = r_new - 1
                uold = mid_inv[i + 1][0]
                lold = mid_inv[i + 1][1]
                erow = cry1[imax1, :].reshape(1, -1)                 # 1 x r_old
                ecol = cre1[ilocl[i + 1][:r_old], 0].reshape(-1, 1)  # r_old x 1
                eel = float(cre1[imax1, 0])
                ecol = lold @ ecol
                erow = erow @ uold
                eel = eel - float((erow @ ecol)[0, 0])
                ecol = uold @ ecol
                erow = erow @ lold

                newU = np.zeros((r_new, r_new))
                newU[:r_old, :r_old] = uold
                newU[:r_old, r_old] = (-ecol / eel).ravel()
                newU[r_old, r_old] = 1.0 / eel
                newL = np.zeros((r_new, r_new))
                newL[:r_old, :r_old] = lold
                newL[r_old, :r_old] = (-erow).ravel()
                newL[r_old, r_old] = 1.0
                mid_inv[i + 1][0] = newU
                mid_inv[i + 1][1] = newL

                y[i] = _rs(y[i], (ry[i], n[i], ry[i + 1]))
                y[i + 1] = _rs(y[i + 1], (ry[i + 1], n[i + 1], ry[i + 2]))

        if use_aux:
            for j in range(Raux):
                phiauxl[i + 1][j] = craux1[j][ilocl[i + 1], :]
                phiauxr[i + 1][j] = craux2[j][:, ilocr[i + 1]]

        i += dir_
        if i == d - 1 or i == -1:
            if verb > 0:
                if xtru is not None:
                    ytest = formtensor(y, mid_inv, d, ry, n)
                    err = _tt_rel_err(ytest, xtru)
                    print("=greedy_cross= swp=%d, max_dx=%.3e, max_rank=%d, "
                          "cum#evals=%d, err_tru=%.3e"
                          % (swp, max_dx, int(np.max(ry)), evalcnt, err))
                else:
                    print("=greedy_cross= swp=%d, max_dx=%.3e, max_rank=%d, "
                          "cum#evals=%d"
                          % (swp, max_dx, int(np.max(ry)), evalcnt))
            if max_dx < tol_exit:
                break
            max_dx = 0.0
            i = 0
            swp += 1

    y = formtensor(y, mid_inv, d, ry, n)
    return y, Jyl, Jyr, ilocl, ilocr, evalcnt


# ---------------------------------------------------------------------------
# Convenience helpers for working with the "list of cores" TT format
# ---------------------------------------------------------------------------
def tt_at(cores, idx):
    """Evaluate a TT tensor (list of cores) at a single 0-based multi-index."""
    v = cores[0][:, idx[0], :]
    for k in range(1, len(cores)):
        v = v @ cores[k][:, idx[k], :]
    return float(v[0, 0])


def tt_full(cores):
    """Expand a TT tensor to a dense NumPy array (small tensors only)."""
    d = len(cores)
    c = cores[0]                       # (1, n0, r1)
    full = c.reshape(cores[0].shape[1], cores[0].shape[2], order="F")
    n = [cores[0].shape[1]]
    for k in range(1, d):
        r = cores[k].shape[0]
        ck = _rs(cores[k], (r, cores[k].shape[1] * cores[k].shape[2]))
        full = full @ ck               # (prod n_<k, n_k * r_{k+1})
        n.append(cores[k].shape[1])
        full = _rs(full, (int(np.prod(n)), cores[k].shape[2]))
    return _rs(full, n)


def _tt_rel_err(a, b):
    """Relative Frobenius error between two small TT tensors."""
    fa = tt_full(a).ravel()
    fb = tt_full(b).ravel()
    return float(np.linalg.norm(fa - fb) / np.linalg.norm(fb))
