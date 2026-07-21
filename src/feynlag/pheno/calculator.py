"""Main orchestrator for decay width and branching ratio calculations."""

import sympy as sp
from feynlag.parameters import ExternalParameter, InternalParameter

class DecayCalculator:
    """Calculates decay widths and branching ratios from a feynlag Model."""
    
    def __init__(self, model):
        self.model = model
        self.numeric_values = {}
        
    def evaluate_parameters(self, external_values):
        """
        Populate numeric_values by evaluating InternalParameters 
        and using the provided external_values dict.
        """
        self.numeric_values.update(external_values)
        
        # Traverse the topological order of parameters
        # and evaluate them to floats.
        for param in self.model.parameters().dependency_order():
            if isinstance(param, ExternalParameter):
                if param.name not in self.numeric_values:
                    self.numeric_values[param.name] = float(param.value)
            elif isinstance(param, InternalParameter):
                expr = param.expr
                if expr is not None:
                    # Substitute known numeric values
                    val = expr.subs(self.numeric_values).evalf()
                    self.numeric_values[param.name] = float(val)

    def calculate_widths(self, parent_field):
        """
        Finds all 2-body decay channels for the given parent field
        and calculates their partial widths.
        """
        widths = {}
        # This will be implemented fully by iterating over self.model.vertices()
        # and using amplitudes.py and kinematics.py
        return widths
