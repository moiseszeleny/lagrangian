"""Discrete symmetries: Z_N and S_3 (extensible to other finite groups).

A discrete group needs only (a) irrep labels with finite generator matrices
and (b) the assignment of field multiplets to irreps.  Invariance of a term
under the whole group is equivalent to invariance under the generators.

Multiplets may span several :class:`~feynlag.fields.Field` objects — e.g. an
S₃ doublet of Higgs SU(2) doublets ``(H1, H2)``: the generator mixes the two
fields component-by-component (each gauge component transforms in parallel).
"""

import sympy as sp

from .base import SymmetryGroup

__all__ = ["DiscreteSymmetry", "ZN", "S3"]


def _component_list(obj):
    """Component symbols of a Field, or a bare Symbol wrapped in a list."""
    if hasattr(obj, "components"):
        return list(obj.components)
    if isinstance(obj, sp.Symbol):
        return [obj]
    raise TypeError(f"expected a Field or Symbol, got {type(obj)}")


def _is_fermion(obj):
    from ..fields import Fermion
    return isinstance(obj, Fermion)


class DiscreteSymmetry(SymmetryGroup):
    """Base class: irreps defined by generator matrices.

    Subclasses populate ``self._irrep_generators``:
    ``{irrep_label: [Matrix, ...]}`` — one matrix per group generator, of
    dimension ``dim(irrep)``.
    """

    def __init__(self, name):
        super().__init__(name)
        self._irrep_generators = {}
        #: list of (irrep_label, tuple_of_multiplet_members)
        self.assignments = []

    @property
    def irreps(self):
        return list(self._irrep_generators)

    def rep_dim(self, irrep):
        return self._irrep_generators[irrep][0].shape[0]

    def assign(self, irrep, *multiplet):
        """Assign a field multiplet to an irrep.

        ``multiplet`` has ``dim(irrep)`` members (Fields or Symbols); members
        of a >1-dimensional multiplet must have matching component structure.

        Example: ``S3.assign('2', H1, H2)`` — (H1, H2) is an S₃ doublet.
        """
        if irrep not in self._irrep_generators:
            raise ValueError(f"{self.name} has no irrep {irrep!r}; "
                             f"available: {self.irreps}")
        dim = self.rep_dim(irrep)
        if len(multiplet) != dim:
            raise ValueError(f"irrep {irrep!r} of {self.name} needs "
                             f"{dim} member(s), got {len(multiplet)}")
        lengths = {len(_component_list(m)) for m in multiplet}
        if len(lengths) != 1:
            raise ValueError("multiplet members must have matching "
                             "component structure")
        fermion_flags = {_is_fermion(m) for m in multiplet}
        if len(fermion_flags) != 1:
            raise ValueError(
                "a discrete multiplet cannot mix Fermion and non-Fermion "
                "members (their components transform via structurally "
                "different substitution mechanisms — Indexed vs. plain "
                "Symbol — so a mixed assignment would silently drop "
                "invariance checking for one side)")
        self.assignments.append((irrep, tuple(multiplet)))
        return self

    def generator_maps(self):
        """Substitution dicts, one per group generator, over all assignments.

        Map ``g``: for each multiplet ``(m₁…m_d)`` in irrep ``r`` with
        generator matrix ``M = ρ_r(g)``, each component slot ``j`` transforms
        as ``m_i[j] → Σ_k M[i,k] m_k[j]``.  ``PartialMu(m_i[j])`` gets the
        identical linear map (discrete generators are spacetime-constant, so
        ``∂_μ`` commutes with them exactly) — this lets
        ``feynlag.invariance.check_discrete_invariance`` handle
        ``Dmu``-built kinetic terms without differentiating through the
        opaque ``PartialMu`` wrapper.

        Fermion multiplets are skipped here (their components are
        ``IndexedBase``, never used as bare atoms — only as
        ``Indexed(base, flavor)`` inside ``Bilinear``\\ s); they go through
        :meth:`fermion_generator_data` instead.
        """
        from ..operators import PartialMu

        n_gens = len(next(iter(self._irrep_generators.values())))
        maps = [dict() for _ in range(n_gens)]
        for irrep, multiplet in self.assignments:
            if _is_fermion(multiplet[0]):
                continue
            comp_lists = [_component_list(m) for m in multiplet]
            n_slots = len(comp_lists[0])
            for gen_index, M in enumerate(self._irrep_generators[irrep]):
                for i in range(len(multiplet)):
                    for j in range(n_slots):
                        image = sum(M[i, k] * comp_lists[k][j]
                                    for k in range(len(multiplet)))
                        dimage = sum(M[i, k] * PartialMu(comp_lists[k][j])
                                    for k in range(len(multiplet)))
                        maps[gen_index][comp_lists[i][j]] = image
                        maps[gen_index][PartialMu(comp_lists[i][j])] = dimage
        return maps

    def components(self):
        """All component symbols across every assigned multiplet."""
        return [c for _, multiplet in self.assignments
                for m in multiplet for c in _component_list(m)]

    def fermion_generator_data(self):
        """Per-generator transform data for fermion multiplets.

        Discrete transformations are *finite* (not infinitesimal), so a
        multiplet member substitutes directly: ``ψ'_i = Σ_k M[i,k] ψ_k``
        (the same formula :meth:`generator_maps` uses for bosons).  Fermion
        components are ``IndexedBase``-typed and only ever appear as
        ``Indexed(base, flavor)``, so this can't be a flat substitution
        dict (the flavor index on the matched node must be preserved) —
        instead it returns, per generator, a dict::

            {IndexedBase: (parallel_list_of_IndexedBase_at_that_slot,
                           local_index_i, transform_matrix)}

        which a ``.replace()``-based builder consumes (see
        ``feynlag.invariance._fermion_transform_discrete``).

        The bar leg uses ``X = (M⁻¹)ᵀ`` rather than ``M``: for ``ψ̄Γψ``
        (summed diagonally over the multiplet index) to be invariant under
        ``ψ_i → Σ_k M[i,k]ψ_k``, requiring
        ``Σ_i ψ̄'_iψ'_i = Σ_i ψ̄_iψ_i`` for the substitution
        ``ψ̄'_i = Σ_k X[i,k]ψ̄_k`` forces ``Σ_i X[i,k]M[i,l] = δ_{kl}``,
        i.e. ``XᵀM = I``, i.e. ``X = (M⁻¹)ᵀ`` (verified directly, not just
        asserted).  Storing ``Xᵀ``'s transpose here means the *same*
        ``Mat[i,k]`` access pattern the field-leg builder uses also gives
        the correct bar-leg coefficient.  ``X`` only coincides with ``M``
        itself when ``M`` is real orthogonal (true for ``S3``'s irreps, not
        for ``ZN``'s complex-phase irreps) — computed generically via
        ``M.inv().T`` rather than assumed.
        """
        n_gens = len(next(iter(self._irrep_generators.values())))
        data = [dict() for _ in range(n_gens)]
        for irrep, multiplet in self.assignments:
            if not _is_fermion(multiplet[0]):
                continue
            comp_lists = [list(m.components) for m in multiplet]
            bar_lists = [list(m.bar_components) for m in multiplet]
            n_slots = len(comp_lists[0])
            d = len(multiplet)
            for gen_index, M in enumerate(self._irrep_generators[irrep]):
                X = M.inv().T
                for i in range(d):
                    for j in range(n_slots):
                        comp_at_slot = [comp_lists[k][j] for k in range(d)]
                        bar_at_slot = [bar_lists[k][j] for k in range(d)]
                        data[gen_index][comp_lists[i][j]] = (
                            comp_at_slot, i, M)
                        data[gen_index][bar_lists[i][j]] = (
                            bar_at_slot, i, X)
        return data


class ZN(DiscreteSymmetry):
    """Cyclic group Z_N: irrep ``k`` acts as ``φ → ω^k φ``, ``ω = e^{2πi/N}``.

    Assign with the integer charge: ``Z2.assign(1, H2)`` for ``H2 → -H2``.
    """

    def __init__(self, name, N):
        super().__init__(name)
        self.N = int(N)
        omega = sp.exp(2 * sp.pi * sp.I / self.N)
        for k in range(self.N):
            self._irrep_generators[k] = [sp.Matrix([[omega ** k]])]


class S3(DiscreteSymmetry):
    """The permutation group S₃ with irreps ``'1'``, ``'1p'``, ``'2'``.

    Generators: ``a`` = 3-cycle, ``b`` = transposition.  The 2-dimensional
    irrep uses the real orthogonal basis:
    ``ρ(a) = R(2π/3)``, ``ρ(b) = diag(1, -1)``.

    Clebsch–Gordan for doublets ``x = (x₁,x₂)``, ``y = (y₁,y₂)`` in this
    basis (2⊗2 = 1 ⊕ 1' ⊕ 2), available via :meth:`doublet_product`.
    """

    def __init__(self, name="S3"):
        super().__init__(name)
        c, s = sp.Rational(-1, 2), sp.sqrt(3) / 2
        rot = sp.Matrix([[c, -s], [s, c]])       # rotation by 2π/3
        ref = sp.Matrix([[1, 0], [0, -1]])       # reflection
        self._irrep_generators = {
            "1":  [sp.Matrix([[1]]), sp.Matrix([[1]])],
            "1p": [sp.Matrix([[1]]), sp.Matrix([[-1]])],
            "2":  [rot, ref],
        }

    @staticmethod
    def doublet_product(x, y):
        """CG decomposition of ``2 ⊗ 2`` in the real orthogonal basis.

        Args:
            x, y: 2-element sequences (components of the two doublets).

        Returns:
            dict with keys ``'1'``, ``'1p'``, ``'2'``:
            - ``'1'``  : ``x₁y₁ + x₂y₂`` (invariant),
            - ``'1p'`` : ``x₁y₂ − x₂y₁``,
            - ``'2'``  : ``(x₁y₁ − x₂y₂, −(x₁y₂ + x₂y₁))`` (doublet).

        The doublet component follows from ``D = conj(w·v)`` with
        ``w = x₁ + i x₂``, ``v = y₁ + i y₂``: under the 2π/3 rotation
        ``w → e^{iθ} w`` one gets ``D → e^{2πi/3} D``, and under the
        reflection ``D → conj(D)`` — exactly the (ρ(a), ρ(b)) action on
        ``(Re D, Im D)``.
        """
        x1, x2 = x
        y1, y2 = y
        return {
            "1": x1 * y1 + x2 * y2,
            "1p": x1 * y2 - x2 * y1,
            "2": (x1 * y1 - x2 * y2, -(x1 * y2 + x2 * y1)),
        }
