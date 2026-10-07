"""Eigenvector-based secant cut (Lan, Bruckner & Lomuscio AAAI'23, eq. 16b):

    s_{i,j} <= (l + u) * t_{i,j} - l * u

where, for a fixed ReLU neuron (layer i, neuron j) and its negative-eigenvector
direction v = (v_xi, v_out) (see eigencuts.py):

    t_{i,j} = v_xi . x_i + v_out * x_{i+1,j}             (linear in P_i)
    s_{i,j} = v^T P_i v = v_xi^T P_i[x_i,x_i] v_xi
              + 2*v_out * (v_xi . P_i[x_i, x_{i+1,j}])
              + v_out^2 * P_i[x_{i+1,j}, x_{i+1,j}]        (quadratic form in P_i)

and [l, u] are the current branch's bounds on t_{i,j}. Both s and t are *linear*
functionals of the existing P_i matrix entries -- this adds no new SDP variable, only
new linear constraints, expressed with the same handler primitives
(add_linear_variable / add_quad_variable / add_bound) used throughout
solve/sdp_solve/SDPmodels/*.py.

Coefficient convention (checked against handler/variable_elements.py
add_dict_quad_to_elements/add_dict_linear_to_elements, called from
handler/mosek_classic/constraints_classic.py's add_var with dividing_non_diag=True,
and verified experimentally -- see sdp_bab/README.md "eigenvector-secant indexing"):
for any off-diagonal pair, add_var DIVIDES the passed value by 2 before storing, and
is_feasible/value_solution MULTIPLY by 2 when reading it back -- these cancel exactly,
so add_quad_variable(value=c, ...) nets out to "contributes exactly c * P[a,b]" for
EITHER call order, with NO implicit doubling. Consequently, to build the full
expansion v^T P v = sum_a v_a^2 P[a,a] + sum_{a!=b} v_a v_b P[a,b], a diagonal term
(a==b) needs value=v_a**2 (single call), and an off-diagonal unordered pair {a,b}
(a!=b) -- which the double sum counts as TWO equal terms v_a*v_b*P[a,b] +
v_b*v_a*P[b,a] -- needs value=2*v_a*v_b in a SINGLE call (not v_a*v_b, and not two
separate calls of v_a*v_b each -- either of those undercounts by half).

x_i and x_{i+1} live in the same chordal block P_i (layer groups are [i, i+1]): layer i
is pinned to front_of_matrix=True (select the group where i is the *first* layer) and
layer i+1 to front_of_matrix=False (select the group where i+1 is the *last* layer) --
both resolve to P_i. See sdp_bab/README.md for why this specific choice is required
(the default inference in verify_variable_z would NOT put both layers in the same block).
"""
import mosek

from solve.sdp_solve.handler.constraints import ConstraintRole
from solve.sdp_solve.handler.variable_elements import ElementsinConstraintsObjectives


def _add_s_terms(constraints_obj, i, j, v_xi, v_out, front_xi, front_out):
    """Shared by add_eigenvector_secant_cut and eval_eigencut_expression_at_solution:
    accumulates s_{i,j} = v^T P_i v into whatever constraint constraints_obj currently
    has open (constraints_obj is self.handler.Constraints either way)."""
    n_i = len(v_xi)
    for a in range(n_i):
        if v_xi[a] == 0:
            continue
        # Diagonal (b == a): single occurrence, no doubling.
        constraints_obj.add_quad_variable(
            var1="z", layer1=i, neuron1=a, front_of_matrix1=front_xi,
            var2="z", layer2=i, neuron2=a, front_of_matrix2=front_xi,
            value=v_xi[a] ** 2,
        )
        # Off-diagonal within the x_i block (b < a): true coefficient is 2*v_a*v_b.
        for b in range(a):
            if v_xi[b] == 0:
                continue
            constraints_obj.add_quad_variable(
                var1="z", layer1=i, neuron1=a, front_of_matrix1=front_xi,
                var2="z", layer2=i, neuron2=b, front_of_matrix2=front_xi,
                value=2 * v_xi[a] * v_xi[b],
            )
    if v_out != 0:
        for a in range(n_i):
            if v_xi[a] == 0:
                continue
            constraints_obj.add_quad_variable(
                var1="z", layer1=i, neuron1=a, front_of_matrix1=front_xi,
                var2="z", layer2=i + 1, neuron2=j, front_of_matrix2=front_out,
                value=2 * v_out * v_xi[a],
            )
        constraints_obj.add_quad_variable(
            var1="z", layer1=i + 1, neuron1=j, front_of_matrix1=front_out,
            var2="z", layer2=i + 1, neuron2=j, front_of_matrix2=front_out,
            value=v_out ** 2,
        )


def _add_t_terms(constraints_obj, i, j, v_xi, v_out, front_xi, front_out, scale=1.0):
    n_i = len(v_xi)
    for a in range(n_i):
        if v_xi[a] == 0:
            continue
        constraints_obj.add_linear_variable(
            var="z", layer=i, neuron=a, front_of_matrix=front_xi, value=scale * v_xi[a],
        )
    if v_out != 0:
        constraints_obj.add_linear_variable(
            var="z", layer=i + 1, neuron=j, front_of_matrix=front_out, value=scale * v_out,
        )


def add_eigenvector_secant_cut(self, eigencut, bounds):
    """eigencut: an eigencuts.EigenCut. bounds: (l, u) tuple, the current branch's bounds
    on t_{i,j} = v . x. If l is None or u is None (neuron not yet bounded / excluded from
    this node's branch set), the cut is skipped -- it is only ever a valid relaxation of
    an already-present constraint, never required for correctness."""
    l, u = bounds
    if l is None or u is None:
        return
    i, j = eigencut.layer, eigencut.neuron
    v_xi, v_out = eigencut.v_xi, eigencut.v_out
    front_xi, front_out = True, False  # layer i: first layer of P_i; layer i+1: last layer of P_i

    C = self.handler.Constraints
    if C.new_constraint(
        f"EigenSecant: s_{i},{j} <= (l+u) t_{i},{j} - l*u  [node bounds l={l:.6g} u={u:.6g}]",
        label="to_change",
    ):
        return

    _add_s_terms(C, i, j, v_xi, v_out, front_xi, front_out)
    _add_t_terms(C, i, j, v_xi, v_out, front_xi, front_out, scale=-(l + u))
    C.add_bound(bound_type=mosek.boundkey.up, bound=-l * u)


def add_all_eigenvector_secant_cuts(self):
    """Adds one cut per (eigencut, bounds) pair in self.eigencuts_bounds -- see
    sdp_bab_model.LayerSDPBaBNode. self.eigencuts is the fixed list of
    eigencuts.EigenCut (network-level, precomputed once); self.eigencuts_bounds maps
    (layer, neuron) -> (l, u) for the neurons active in this B&B node (others are
    omitted, meaning the vanilla layer SDP relaxation is used for them -- always a
    valid, just looser, constraint)."""
    for cut in self.eigencuts:
        bounds = self.eigencuts_bounds.get((cut.layer, cut.neuron))
        if bounds is None:
            continue
        self.add_eigenvector_secant_cut(cut, bounds)


def eval_eigencut_expression_at_solution(self, eigencut, X):
    """Evaluates t_{i,j} and s_{i,j} (see module docstring) at a solved node's primal
    matrices X (dict num_matrix -> dense ndarray, e.g. from get_solution()), by
    building the SAME element list add_eigenvector_secant_cut would (same
    add_linear_variable/add_quad_variable calls, via _add_s_terms/_add_t_terms) into a
    scratch Constraints entry, then reading back elements.decode_key_vec() (i, j,
    num_matrix, value arrays -- the same decoding handler.constraints.formate_cstr()
    uses) and summing value * (2 if i!=j else 1) * X[num_matrix][i][j] -- the same
    read-side rule used by handler.value_solution/is_feasible, which cancels the
    write-side dividing_non_diag halving (see module docstring). The scratch
    constraint is popped before returning -- it never reaches add_bound(), so it is
    never sent to MOSEK.

    Returns (t_value, s_value, violation) with violation = |s_value - t_value**2|
    (Algorithm 1's d^r_{i,j}).
    """
    i, j = eigencut.layer, eigencut.neuron
    v_xi, v_out = eigencut.v_xi, eigencut.v_out
    front_xi, front_out = True, False
    C = self.handler.Constraints

    def _probe(name, build_fn):
        """Bypasses Constraints.new_constraint()/check_current_constraint(): by the
        time a solved node is evaluated, end_constraints() has already advanced
        current_num_constraint past the end of list_cstr (see
        Targeted_SDP._add_extra_constraints_hook's docstring), so new_constraint()'s
        own pre-check would IndexError. Appends/pops a minimal well-formed entry
        directly instead -- same dict shape as new_constraint(), never touches MOSEK
        (no add_bound() call, so add_to_task() never sees it)."""
        C.list_cstr.append({
            "name": name,
            "elements": ElementsinConstraintsObjectives(C.indexes_variables.max_index),
            "constant": 0.0, "lb": None, "ub": None, "bound_type": None,
            "dual_value": None, "label": "to_change", "is_quadratic": False,
            "not_in_miqcr": False, "role": ConstraintRole.HARD, "dualizable_family": None,
        })
        saved_current = C.current_num_constraint
        C.current_num_constraint = len(C.list_cstr) - 1
        build_fn()
        value = _eval_elements(C.list_cstr[-1]["elements"], X)
        C.list_cstr.pop()
        C.current_num_constraint = saved_current
        return value

    s_value = _probe(f"__probe_eigencut_s_{i}_{j}__",
                      lambda: _add_s_terms(C, i, j, v_xi, v_out, front_xi, front_out))
    t_value = _probe(f"__probe_eigencut_t_{i}_{j}__",
                      lambda: _add_t_terms(C, i, j, v_xi, v_out, front_xi, front_out, scale=1.0))

    return t_value, s_value, abs(s_value - t_value ** 2)


def _eval_elements(elements, X):
    i_arr, j_arr, num_matrix_arr, val_arr = elements.decode_key_vec()
    total = 0.0
    for k in range(len(val_arr)):
        num_matrix, i_row, j_col, coeff = int(num_matrix_arr[k]), int(i_arr[k]), int(j_arr[k]), float(val_arr[k])
        if i_row != j_col:
            coeff *= 2.0
        total += coeff * X[num_matrix][i_row][j_col]
    return total
