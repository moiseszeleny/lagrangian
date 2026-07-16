"""Gauge and discrete group tests."""

import sympy as sp
import pytest

from feynlag import S3, SU2, SU3, U1, WeylFermion, ZN, structure_constants


class TestSU2:
    def test_fundamental_algebra(self):
        """[T^a, T^b] = i eps_abc T^c."""
        g = SU2("SU2L")
        T = g.generators(2)
        for a in range(3):
            for b in range(3):
                comm = T[a] * T[b] - T[b] * T[a]
                expected = sum((sp.I * sp.LeviCivita(a + 1, b + 1, c + 1) * T[c]
                                for c in range(3)), sp.zeros(2, 2))
                assert sp.simplify(comm - expected) == sp.zeros(2, 2)

    def test_adjoint_algebra(self):
        g = SU2("SU2L")
        T = g.generators(3)
        for a in range(3):
            for b in range(3):
                comm = T[a] * T[b] - T[b] * T[a]
                expected = sum((sp.I * sp.LeviCivita(a + 1, b + 1, c + 1) * T[c]
                                for c in range(3)), sp.zeros(3, 3))
                assert sp.simplify(comm - expected) == sp.zeros(3, 3)

    def test_normalization(self):
        """Tr(T^a T^b) = delta_ab / 2 in the fundamental."""
        T = SU2("SU2L").generators(2)
        for a in range(3):
            for b in range(3):
                tr = sp.trace(T[a] * T[b])
                assert tr == (sp.Rational(1, 2) if a == b else 0)


class TestSU3:
    def test_fundamental_normalization(self):
        T = SU3("SU3c").generators(3)
        for a in range(8):
            for b in range(8):
                tr = sp.simplify(sp.trace(T[a] * T[b]))
                assert tr == (sp.Rational(1, 2) if a == b else 0)

    def test_bad_rep_raises(self):
        with pytest.raises(ValueError):
            SU3("SU3c").rep_dim(5)

    def test_adjoint_algebra(self):
        """[T^a, T^b] = i f^abc T^c, cross-checked against the independently
        coded structure_constants(group) (yangmills.py), not just re-deriving
        gauge.py's own inline f from itself."""
        g = SU3("SU3c")
        T = g.generators(8)
        f = structure_constants(g)
        for a in range(8):
            for b in range(8):
                comm = T[a] * T[b] - T[b] * T[a]
                expected = sum(
                    (sp.I * f.get((a, b, c), sp.S.Zero) * T[c]
                     for c in range(8)), sp.zeros(8, 8))
                assert sp.simplify(comm - expected) == sp.zeros(8, 8)

    def test_adjoint_normalization(self):
        """Tr(T^a T^b) = C_A delta_ab = N delta_ab in the adjoint of SU(N)."""
        T = SU3("SU3c").generators(8)
        for a in range(8):
            for b in range(8):
                tr = sp.simplify(sp.trace(T[a] * T[b]))
                assert tr == (3 if a == b else 0)


class TestU1:
    def test_charge_generator(self):
        q = sp.Rational(1, 2)
        gens = U1("U1Y").generators(q)
        assert gens == [sp.Matrix([[q]])]


class TestZN:
    def test_z2_generator_map(self):
        Z2 = ZN("Z2", 2)
        phi = sp.Symbol("phi")
        Z2.assign(1, phi)
        (gmap,) = Z2.generator_maps()
        assert gmap[phi] == -phi

    def test_z3_phase(self):
        Z3 = ZN("Z3", 3)
        phi = sp.Symbol("phi")
        Z3.assign(1, phi)
        (gmap,) = Z3.generator_maps()
        omega = sp.exp(2 * sp.pi * sp.I / 3)
        assert sp.simplify(gmap[phi] - omega * phi) == 0


class TestS3:
    def setup_method(self):
        self.S3 = S3()
        self.x1, self.x2 = sp.symbols("x1 x2", real=True)
        self.y1, self.y2 = sp.symbols("y1 y2", real=True)

    def _transform(self, expr, gen_index, pairs):
        """Apply the irrep-2 generator to doublet component pairs."""
        M = self.S3._irrep_generators["2"][gen_index]
        sub = {}
        for (a1, a2) in pairs:
            sub[a1] = M[0, 0] * a1 + M[0, 1] * a2
            sub[a2] = M[1, 0] * a1 + M[1, 1] * a2
        return expr.xreplace(sub)

    def test_generators_orders(self):
        """rho(a)^3 = 1, rho(b)^2 = 1 in every irrep."""
        for irrep, (a, b) in self.S3._irrep_generators.items():
            n = a.shape[0]
            assert sp.simplify(a**3 - sp.eye(n)) == sp.zeros(n, n)
            assert sp.simplify(b**2 - sp.eye(n)) == sp.zeros(n, n)
            # braid relation (b a)^2 = 1 for S3 = <a, b | a^3 = b^2 = (ba)^2>
            ba = b * a
            assert sp.simplify(ba**2 - sp.eye(n)) == sp.zeros(n, n)

    def test_invariant_singlet_product(self):
        inv = self.S3.doublet_product((self.x1, self.x2),
                                      (self.y1, self.y2))["1"]
        for gi in (0, 1):
            transformed = self._transform(
                inv, gi, [(self.x1, self.x2), (self.y1, self.y2)])
            assert sp.simplify(transformed - inv) == 0

    def test_1p_transforms_with_sign(self):
        onep = self.S3.doublet_product((self.x1, self.x2),
                                       (self.y1, self.y2))["1p"]
        # invariant under the rotation, odd under the reflection
        rot = self._transform(onep, 0, [(self.x1, self.x2), (self.y1, self.y2)])
        assert sp.simplify(rot - onep) == 0
        ref = self._transform(onep, 1, [(self.x1, self.x2), (self.y1, self.y2)])
        assert sp.simplify(ref + onep) == 0

    def test_doublet_product_transforms_as_doublet(self):
        """CG '2' component must transform with the irrep-2 matrices."""
        d1, d2 = self.S3.doublet_product((self.x1, self.x2),
                                         (self.y1, self.y2))["2"]
        for gi in (0, 1):
            M = self.S3._irrep_generators["2"][gi]
            t1 = self._transform(d1, gi, [(self.x1, self.x2),
                                          (self.y1, self.y2)])
            t2 = self._transform(d2, gi, [(self.x1, self.x2),
                                          (self.y1, self.y2)])
            assert sp.simplify(t1 - (M[0, 0] * d1 + M[0, 1] * d2)) == 0
            assert sp.simplify(t2 - (M[1, 0] * d1 + M[1, 1] * d2)) == 0

    def test_assign_wrong_size_raises(self):
        with pytest.raises(ValueError):
            self.S3.assign("2", sp.Symbol("only_one"))

    def test_fermion_generator_data_field_leg(self):
        """Field leg uses M directly: psi'_i = sum_k M[i,k] psi_k."""
        psi1 = WeylFermion("gpsi1", reps={}, chirality="L", nflavors=1,
                           component_names=["gp1"])
        psi2 = WeylFermion("gpsi2", reps={}, chirality="L", nflavors=1,
                           component_names=["gp2"])
        self.S3.assign("2", psi1, psi2)
        p1c, = psi1.components
        p2c, = psi2.components
        idx = sp.Symbol("idx", integer=True)

        data = self.S3.fermion_generator_data()
        for gi in (0, 1):
            M = self.S3._irrep_generators["2"][gi]
            comp_at_slot, i, Mat = data[gi][p1c]
            assert Mat == M and i == 0
            transformed = sum(Mat[i, k] * comp_at_slot[k][idx]
                              for k in range(2))
            expected = M[0, 0] * p1c[idx] + M[0, 1] * p2c[idx]
            assert sp.simplify(transformed - expected) == 0

    def test_fermion_generator_data_bar_leg_uses_Minv_transpose(self):
        """Bar leg uses X=(M^-1)^T, verified against the invariance
        condition sum_i psibar_i psi_i unchanged, not merely M itself
        (only coincides with M for real-orthogonal generators)."""
        psi1 = WeylFermion("bpsi1", reps={}, chirality="L", nflavors=1,
                           component_names=["bp1"])
        psi2 = WeylFermion("bpsi2", reps={}, chirality="L", nflavors=1,
                           component_names=["bp2"])
        self.S3.assign("2", psi1, psi2)
        p1c, p2c = psi1.components[0], psi2.components[0]
        p1bar, p2bar = psi1.bar_components[0], psi2.bar_components[0]
        idx = sp.Symbol("idx", integer=True)

        data = self.S3.fermion_generator_data()
        for gi in (0, 1):
            M = self.S3._irrep_generators["2"][gi]
            field_sub = {}
            bar_sub = {}
            for base, node in ((p1c, p1c[idx]), (p2c, p2c[idx])):
                comp_at_slot, i, Mat = data[gi][base]
                field_sub[node] = sum(Mat[i, k] * comp_at_slot[k][idx]
                                      for k in range(2))
            for base, node in ((p1bar, p1bar[idx]), (p2bar, p2bar[idx])):
                comp_at_slot, i, Mat = data[gi][base]
                assert sp.simplify(Mat - M.inv().T) == sp.zeros(2, 2)
                bar_sub[node] = sum(Mat[i, k] * comp_at_slot[k][idx]
                                    for k in range(2))
            original = p1bar[idx] * p1c[idx] + p2bar[idx] * p2c[idx]
            transformed = (bar_sub[p1bar[idx]] * field_sub[p1c[idx]]
                          + bar_sub[p2bar[idx]] * field_sub[p2c[idx]])
            assert sp.expand(transformed - original) == 0


class TestZNFermion:
    def test_bar_transform_uses_conjugate_phase_not_M(self):
        """For ZN, X=(M^-1)^T=conjugate(M) != M in general — a genuine
        (not merely real-orthogonal) exercise of the derived formula."""
        Z3 = ZN("Z3", 3)
        psi = WeylFermion("zpsi", reps={}, chirality="L", nflavors=1,
                          component_names=["zp"])
        Z3.assign(1, psi)
        pc, = psi.components
        pbar, = psi.bar_components

        data = Z3.fermion_generator_data()
        (gen1,) = data  # ZN has a single generator
        _, _, M_field = gen1[pc]
        _, _, X_bar = gen1[pbar]
        omega = sp.exp(2 * sp.pi * sp.I / 3)
        assert sp.simplify(M_field[0, 0] - omega) == 0
        assert sp.simplify(X_bar[0, 0] - sp.conjugate(omega)) == 0
        assert sp.simplify(M_field[0, 0] - X_bar[0, 0]) != 0  # genuinely differ

    def test_mixed_fermion_scalar_multiplet_rejected(self):
        from feynlag import Scalar, SU2, ExternalParameter
        s3 = S3()
        gw = ExternalParameter("gwx", 0.65, positive=True)
        SU2L = SU2("SU2Lx", coupling=gw)
        H = Scalar("Hx", reps={SU2L: 2}, component_names=["Hpx", "H0x"])
        psi = WeylFermion("mpsi", reps={SU2L: 2}, chirality="L", nflavors=1,
                          component_names=["ma", "mb"])
        with pytest.raises(ValueError, match="cannot mix Fermion"):
            s3.assign("2", H, psi)
