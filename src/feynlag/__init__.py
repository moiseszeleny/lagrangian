"""feynlag — tree-level Feynman rules from BSM Lagrangians in pure SymPy."""

__version__ = "0.1.0"

from .conventions import METRIC_SIGNATURE, SQRT2, tidy
from .dirac import (
    DiracC, DiracGamma, DiracGammaLower, MetricTensor, dirac0, diracC, diracI,
    diracPL, diracPR, dirac_conjugate, gamma_simplify, majorana_symmetry_sign,
    minkowski_metric,
)
from .operators import (
    D_linear, Dmu, PartialMu, expand_derivatives, momentum, to_momentum_space,
)
from .vertices import (
    extract_interaction_coefficients, feynman_rule, vertex_multiplicity,
)
from .vertices.vertex import LORENTZ_CATALOG, Vertex, classify_spins
from .vertices.yangmills import (
    cubic_couplings, quartic_couplings, structure_constants,
)
from .vacuum.masses import gauge_mass_matrix
from .verify import (
    UFORoundTripReport, check_dimension, is_hermitian, is_symmetric,
    numeric_equal, round_trip_reconstruct, seesaw_light_mass,
    verify_ufo_numeric,
)
from .export import latex_feynman_table
from .parameters import (
    ExternalParameter, InternalParameter, Parameter, ParameterSet,
)
from .groups import SU2, SU3, U1, S3, ZN, DiscreteSymmetry, GaugeGroup
from .fields import (
    DiracFermion, Field, Fermion, GaugeBoson, MajoranaFermion, Scalar,
    WeylFermion, bar_partner, conjugate_pair, dag, hc,
)
from .vacuum import (
    Rotation, Vacuum, build_mass_matrix, charged_mass_matrix,
    diagonalize_orthogonal_2x2, diagonalize_svd, diagonalize_svd_2x2,
    diagonalize_takagi,
    rotation_2x2, scalar_mass_matrix, solve_mixing_angle_2x2,
)
from .vertices.bilinear import (
    Bilinear, MajoranaBilinear, expand_bilinear, extract_fermion_vertices,
    extract_majorana_vertices, fermion_feynman_rule, fermion_gauge_current,
    fermion_mass_matrix, four_fermion_feynman_rule, majorana_feynman_rule,
    majorana_mass_matrix,
)
from .invariance import (
    check_discrete_invariance, check_gauge_invariance, check_hermiticity,
    check_mass_dimension, gauge_variation,
)
from .lagrangian import (
    InvarianceReport, Lagrangian, LagrangianTerm, Model, ValidationReport,
)
from .anomalies import (
    AnomalyReport, anomaly_coefficients, check_anomaly_free,
)
from .charges import (
    ChargeConservationReport, ChargeConsistencyReport, ChargeRegistry,
    HermiticityPairingReport, check_charge_conservation,
    check_charge_consistency, check_hermiticity_pairing,
    derive_charge_operator, physical_charges,
)
from .flavor import CKM_ELEMENT_NAMES, cabibbo_2x2, standard_ckm
from .suggest import (
    SuggestedTerm, build_lagrangian, suggest_kinetic, suggest_potential,
    suggest_yukawa,
)

__all__ = [
    "Parameter", "ExternalParameter", "InternalParameter", "ParameterSet",
    "GaugeGroup", "U1", "SU2", "SU3", "DiscreteSymmetry", "ZN", "S3",
    "Field", "Scalar", "Fermion", "WeylFermion", "DiracFermion",
    "MajoranaFermion", "GaugeBoson", "dag", "hc", "conjugate_pair",
    "bar_partner",
    "Vacuum", "Rotation", "rotation_2x2", "solve_mixing_angle_2x2",
    "diagonalize_orthogonal_2x2", "diagonalize_svd", "diagonalize_svd_2x2",
    "diagonalize_takagi",
    "build_mass_matrix", "scalar_mass_matrix", "charged_mass_matrix",
    "Bilinear", "MajoranaBilinear", "expand_bilinear",
    "extract_fermion_vertices", "extract_majorana_vertices",
    "fermion_feynman_rule", "fermion_gauge_current", "fermion_mass_matrix",
    "four_fermion_feynman_rule", "majorana_mass_matrix",
    "majorana_feynman_rule",
    "gauge_variation", "check_gauge_invariance", "check_discrete_invariance",
    "check_hermiticity", "check_mass_dimension",
    "Lagrangian", "LagrangianTerm", "Model", "InvarianceReport",
    "ValidationReport",
    "METRIC_SIGNATURE", "SQRT2", "tidy",
    "DiracGamma", "DiracGammaLower", "DiracC", "MetricTensor", "dirac0",
    "diracI", "diracPL", "diracPR", "diracC", "dirac_conjugate",
    "gamma_simplify", "majorana_symmetry_sign", "minkowski_metric",
    "D_linear", "Dmu", "PartialMu", "expand_derivatives", "momentum",
    "to_momentum_space",
    "Vertex", "classify_spins", "LORENTZ_CATALOG",
    "structure_constants", "cubic_couplings", "quartic_couplings",
    "gauge_mass_matrix",
    "extract_interaction_coefficients", "feynman_rule", "vertex_multiplicity",
    "check_dimension", "is_hermitian", "is_symmetric", "numeric_equal",
    "round_trip_reconstruct", "seesaw_light_mass",
    "UFORoundTripReport", "verify_ufo_numeric",
    "latex_feynman_table",
    "SuggestedTerm", "suggest_potential", "suggest_yukawa", "suggest_kinetic",
    "build_lagrangian",
    "AnomalyReport", "anomaly_coefficients", "check_anomaly_free",
    "ChargeRegistry", "ChargeConservationReport", "ChargeConsistencyReport",
    "HermiticityPairingReport", "check_charge_conservation",
    "check_charge_consistency", "check_hermiticity_pairing",
    "derive_charge_operator", "physical_charges",
    "standard_ckm", "cabibbo_2x2", "CKM_ELEMENT_NAMES",
]
