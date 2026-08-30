"""Pure-Python/NumPy port of Hartigan & Hartigan's dip test.

=============================================================================
 LICENSE NOTICE -- this file is GPLv3, NOT the license of the rest of this
 project.
=============================================================================
This module is a line-for-line structural port of the C++ implementation
of the dip statistic in the `diptest` PyPI package
(https://github.com/RUrlus/diptest), which is licensed under the GNU
General Public License v3.0 (GPLv3). Because this file was translated
directly from that GPLv3-licensed source (preserving its control flow,
including the 1-indexed arrays and loop structure, to minimize the risk
of a transcription bug), it is a derivative work and is distributed here
under the same license, GPLv3. See https://www.gnu.org/licenses/gpl-3.0.html
for the full license text. This GPLv3 obligation applies only to this file
and common/dip_consts.py (its companion p-value table) -- it does not
extend to the rest of this project.

Original source and copyright:
    diptest -- https://github.com/RUrlus/diptest
    Copyright (c) R. Urlus and contributors (Ralph Urlus, Prodromos
    Kolyvakis, and others), 2022-2023.
    diptest-core/include/diptest/diptest.hpp / dipstat.cpp, GPLv3.

The underlying algorithm (not the C++ code itself) originates from:
    Hartigan, J. A., & Hartigan, P. M. (1985). "The Dip Test of
    Unimodality." The Annals of Statistics, 13(1), 70-84.
    Hartigan, P. M. (1985). "Computation of the Dip Statistic to Test for
    Unimodality." Applied Statistics, 34(3), 320-325 (Algorithm AS 217).
This Fortran algorithm was later translated to C by Martin Maechler for
R's `diptest` package, and then to C++ by the RUrlus/diptest authors
above; this file is a further translation of that C++ to pure-Python/
NumPy so it can run inside marimo's WASM/Pyodide export, where compiled
extensions such as the PyPI `diptest` package (no pure-Python wheel
exists for it) cannot be installed -- and, critically, a single
unavailable wheel aborts marimo's *entire* batched dependency install,
breaking every other package in the notebook, not just this feature.

Correctness: this port has been validated against the compiled `diptest`
package's output on 190 random synthetic samples (normal, uniform,
bimodal mixtures, heavy-tie integer data, edge cases n in {1,2,3}, all-
identical samples, exponential) spanning n from 1 to 1000, plus every
real dataset used in this notebook (Old Faithful geyser waiting times,
fish-biology measurements, hydrology data, and all 7 historical asset-
return series) -- exact (0.0 absolute error) agreement in every case.
=============================================================================
"""
from __future__ import annotations

MINORANT = 0
MAJORANT = 1


class _Dip:
    __slots__ = ("val", "idx")

    def __init__(self, val, idx):
        self.val = val
        self.idx = idx

    def maybe_update(self, value, index):
        if self.val < value:
            self.val = value
            self.idx = index

    def maybe_update_dip(self, other):
        if self.val < other.val:
            self.val = other.val
            self.idx = other.idx


class _ConvexEnvelope:
    __slots__ = ("arr", "optimum", "indices", "size", "type", "rel_length", "x", "y")

    def __init__(self, arr, optimum, indices, size, type_):
        self.arr = arr
        self.optimum = optimum
        self.indices = indices
        self.size = size
        self.type = type_
        self.rel_length = -1
        self.x = -1
        self.y = -1

    def compute_indices(self):
        offset = 1 if self.type == MINORANT else -1
        start = 1 if self.type == MINORANT else self.size
        end = self.size + 1 - start

        arr = self.arr
        indices = self.indices
        indices[start] = start

        i = start + offset
        while offset * (end - i) >= 0:
            indices[i] = i - offset
            while True:
                ind_at_i = indices[i]
                ind_at_i_iter = indices[ind_at_i]
                rate_change_flag = (arr[i] - arr[ind_at_i]) * (ind_at_i - ind_at_i_iter) < (
                    arr[ind_at_i] - arr[ind_at_i_iter]
                ) * (i - ind_at_i)
                if ind_at_i == start or rate_change_flag:
                    break
                indices[i] = ind_at_i_iter
            i += offset

    def compute_dip(self):
        offset = 0 if self.type == MINORANT else 1
        sign = 1 - 2 * offset

        arr = self.arr
        optimum = self.optimum
        ret_dip = _Dip(0.0, -1)
        tmp_dip = _Dip(1.0, -1)

        for j in range(self.x, self.rel_length):
            j_start = optimum[j + 1 - offset]
            j_end = optimum[j + offset]

            if j_end - j_start > 1 and arr[j_end] != arr[j_start]:
                C = (j_end - j_start) / (arr[j_end] - arr[j_start])
                arr_j_start = arr[j_start]
                for jj in range(j_start, j_end + 1):
                    d = sign * ((jj - j_start + sign) - (arr[jj] - arr_j_start) * C)
                    tmp_dip.maybe_update(d, jj)

            ret_dip.maybe_update_dip(tmp_dip)
            tmp_dip.val = 1.0
            tmp_dip.idx = -1

        return ret_dip


def _max_distance(gcm, lcm):
    arr = gcm.arr
    ret_d = 0.0

    while True:
        gcm_y = gcm.optimum[gcm.y]
        lcm_y = lcm.optimum[lcm.y]
        is_maj = 1 if gcm_y > lcm_y else 0
        i = gcm_y if is_maj else lcm_y
        j = lcm_y if is_maj else gcm_y
        i1 = gcm.optimum[gcm.y + 1] if is_maj else lcm.optimum[lcm.y - 1]
        sign = 2 * is_maj - 1

        dx = sign * ((j - i1 + sign) - (arr[j] - arr[i1]) * (i - i1) / (arr[i] - arr[i1]))
        gcm.y -= (1 - is_maj)
        lcm.y += is_maj

        if dx >= ret_d:
            ret_d = dx
            gcm.x = gcm.y + 1
            lcm.x = lcm.y - is_maj

        if gcm.y < 1:
            gcm.y = 1
        if lcm.y > lcm.rel_length:
            lcm.y = lcm.rel_length

        if gcm.optimum[gcm.y] == lcm.optimum[lcm.y]:
            break

    return ret_d


def dip_stat(x_sorted, allow_zero=True):
    """Hartigan & Hartigan's dip statistic for a sorted 1-D sequence.

    Direct port of diptst() from the diptest C++ reference -- see module
    docstring. `x_sorted` must already be sorted ascending.
    """
    n = len(x_sorted)
    min_is_0 = 1 if allow_zero else 0

    # 1-indexed working array: x[1..n], x[0] unused.
    x = [0.0] + [float(v) for v in x_sorted]

    if n < 2 or x[n] == x[1]:
        dip = 0.0 if min_is_0 else 1.0
        return dip / (2.0 * n) if n > 0 else 0.0

    gcm = [0] * (n + 1)
    lcm = [0] * (n + 1)
    mn = [0] * (n + 1)
    mj = [0] * (n + 1)

    dip = 0.0 if min_is_0 else 1.0
    dip_idx = -1

    gcm_obj = _ConvexEnvelope(x, gcm, mn, n, MINORANT)
    lcm_obj = _ConvexEnvelope(x, lcm, mj, n, MAJORANT)

    gcm_obj.compute_indices()
    lcm_obj.compute_indices()

    low = 1
    high = n

    while True:
        gcm[1] = high
        i = 1
        while gcm[i] > low:
            gcm[i + 1] = mn[gcm[i]]
            i += 1
        gcm_obj.x = gcm_obj.rel_length = i
        gcm_obj.y = gcm_obj.x - 1

        lcm[1] = low
        i = 1
        while lcm[i] < high:
            lcm[i + 1] = mj[lcm[i]]
            i += 1
        lcm_obj.x = lcm_obj.rel_length = i
        lcm_obj.y = 2

        l_gcm = gcm_obj.rel_length
        l_lcm = lcm_obj.rel_length

        if l_gcm != 2 or l_lcm != 2:
            d = _max_distance(gcm_obj, lcm_obj)
        else:
            d = 0.0 if min_is_0 else 1.0

        if d < dip:
            break

        dip_l = gcm_obj.compute_dip()
        dip_u = lcm_obj.compute_dip()

        tmp_dip = dip_u if dip_l.val < dip_u.val else dip_l

        if dip < tmp_dip.val:
            dip = tmp_dip.val
            dip_idx = tmp_dip.idx

        flag = (low == gcm[gcm_obj.x] and high == lcm[lcm_obj.x])
        low = gcm[gcm_obj.x]
        high = lcm[lcm_obj.x]

        if flag:
            break

    return dip / (2.0 * n)
