"""Static file contents shipped with every generated UFO model.

``OBJECT_LIBRARY`` follows the standard UFO ``object_library.py`` API
(UFOBaseClass, Particle with ``anti()``, Parameter, Vertex, Coupling,
Lorentz, CouplingOrder, and the ``all_*`` registries) so MadGraph-style
tooling and the import smoke test both work.
"""

OBJECT_LIBRARY = '''\
##
## object_library.py — standard UFO object definitions (feynlag export)
##

import cmath
import re


class UFOError(Exception):
    """Exception raised if when inconsistencies are detected in the UFO model."""
    pass


class UFOBaseClass(object):
    """The class from which all FeynRules-style UFO classes are derived."""

    require_args = []

    def __init__(self, *args, **options):
        assert len(self.require_args) == len(args)
        for i, name in enumerate(self.require_args):
            setattr(self, name, args[i])
        for option, value in options.items():
            setattr(self, option, value)

    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)

    def get_all(self):
        """Return a dictionary containing all the information of the object."""
        return self.__dict__

    def __str__(self):
        return self.name

    def nice_string(self):
        """ return string with the full information """
        return '\\n'.join(['%s \\t: %s' % (name, value)
                          for name, value in self.__dict__.items()])

    def __repr__(self):
        replacements = [
            ('+', '__plus__'), ('-', '__minus__'), ('@', '__at__'),
            ('!', '__exclam__'), ('?', '__quest__'), ('*', '__star__'),
            ('~', '__tilde__'),
        ]
        text = self.name
        for orig, sub in replacements:
            text = text.replace(orig, sub)
        return text


all_particles = []


class Particle(UFOBaseClass):
    """A standard Particle"""

    require_args = ['pdg_code', 'name', 'antiname', 'spin', 'color', 'mass',
                    'width', 'texname', 'antitexname', 'charge']

    require_args_all = ['pdg_code', 'name', 'antiname', 'spin', 'color',
                        'mass', 'width', 'texname', 'antitexname', 'charge',
                        'line', 'propagating', 'goldstoneboson']

    def __init__(self, pdg_code, name, antiname, spin, color, mass, width,
                 texname, antitexname, charge, line=None, propagating=True,
                 goldstoneboson=False, **options):
        args = (pdg_code, name, antiname, spin, color, mass, width, texname,
                antitexname, float(charge))
        UFOBaseClass.__init__(self, *args, **options)
        global all_particles
        all_particles.append(self)
        self.propagating = propagating
        self.goldstoneboson = goldstoneboson
        self.selfconjugate = (name == antiname)
        if line is None:
            self.line = self.find_line_type()
        else:
            self.line = line

    def find_line_type(self):
        """ find how we draw a line if not defined
        valid output: dashed/straight/wavy/curly/double/swavy/scurly
        """
        spin = self.spin
        color = self.color
        if spin == 1:
            return 'dashed'
        elif spin == 2:
            if not self.selfconjugate:
                return 'straight'
            elif color == 1:
                return 'swavy'
            else:
                return 'scurly'
        elif spin == 3:
            if color == 1:
                return 'wavy'
            else:
                return 'curly'
        elif spin == 5:
            return 'double'
        elif spin == -1:
            return 'dotted'
        else:
            return 'dashed'  # not supported yet

    def anti(self):
        if self.selfconjugate:
            raise Exception('%s has no anti particle.' % self.name)
        outdic = {}
        for k, v in self.__dict__.items():
            if k not in self.require_args_all:
                outdic[k] = -v
        if self.color in [1, 8]:
            newcolor = self.color
        else:
            newcolor = -self.color
        return Particle(-self.pdg_code, self.antiname, self.name, self.spin,
                        newcolor, self.mass, self.width, self.antitexname,
                        self.texname, -self.charge, self.line,
                        self.propagating, self.goldstoneboson, **outdic)


all_parameters = []


class Parameter(UFOBaseClass):

    require_args = ['name', 'nature', 'type', 'value', 'texname']

    def __init__(self, name, nature, type, value, texname, lhablock=None,
                 lhacode=None):
        args = (name, nature, type, value, texname)
        UFOBaseClass.__init__(self, *args)
        args = (name, nature, type, value, texname)
        global all_parameters
        all_parameters.append(self)
        if (lhablock is None or lhacode is None) and nature == 'external':
            raise Exception('Need LHA information for external parameter '
                            '"%s".' % name)
        self.lhablock = lhablock
        self.lhacode = lhacode


all_vertices = []


class Vertex(UFOBaseClass):

    require_args = ['name', 'particles', 'color', 'lorentz', 'couplings']

    def __init__(self, name, particles, color, lorentz, couplings,
                 **opt):
        args = (name, particles, color, lorentz, couplings)
        UFOBaseClass.__init__(self, *args, **opt)
        args = (particles, color, lorentz, couplings)
        global all_vertices
        all_vertices.append(self)


all_couplings = []


class Coupling(UFOBaseClass):

    require_args = ['name', 'value', 'order']

    def __init__(self, name, value, order, **opt):
        args = (name, value, order)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_couplings
        all_couplings.append(self)


all_lorentz = []


class Lorentz(UFOBaseClass):

    require_args = ['name', 'spins', 'structure']

    def __init__(self, name, spins, structure='external', **opt):
        args = (name, spins, structure)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_lorentz
        all_lorentz.append(self)


all_functions = []


class Function(object):

    def __init__(self, name, arguments, expression):
        global all_functions
        all_functions.append(self)
        self.name = name
        self.arguments = arguments
        self.expr = expression

    def __call__(self, *opt):
        for i, arg in enumerate(self.arguments):
            exec('%s = %s' % (arg, opt[i]))
        return eval(self.expr)


all_orders = []


class CouplingOrder(object):

    def __init__(self, name, expansion_order, hierarchy, perturbative_expansion=0):
        global all_orders
        all_orders.append(self)
        self.name = name
        self.expansion_order = expansion_order
        self.hierarchy = hierarchy
        self.perturbative_expansion = perturbative_expansion


all_decays = []


class Decay(UFOBaseClass):
    require_args = ['particle', 'partial_widths']

    def __init__(self, particle, partial_widths, **opt):
        args = (particle, partial_widths)
        UFOBaseClass.__init__(self, *args, **opt)
        global all_decays
        all_decays.append(self)
        particle.partial_widths = partial_widths


all_propagators = []


class Propagator(object):

    def __init__(self, name, numerator, denominator=None, **opt):
        global all_propagators
        all_propagators.append(self)
        self.name = name
        self.numerator = numerator
        self.denominator = denominator
        for label, value in opt.items():
            setattr(self, label, value)
'''

FUNCTION_LIBRARY = '''\
##
## function_library.py — standard UFO auxiliary functions
##

import cmath
from .object_library import all_functions, Function

complexconjugate = Function(name='complexconjugate',
                            arguments=('z',),
                            expression='z.conjugate()')

re = Function(name='re',
              arguments=('z',),
              expression='z.real')

im = Function(name='im',
              arguments=('z',),
              expression='z.imag')

# New functions (trigonometric)

sec = Function(name='sec',
               arguments=('z',),
               expression='1./cmath.cos(z)')

asec = Function(name='asec',
                arguments=('z',),
                expression='cmath.acos(1./z)')

csc = Function(name='csc',
               arguments=('z',),
               expression='1./cmath.sin(z)')

acsc = Function(name='acsc',
                arguments=('z',),
                expression='cmath.asin(1./z)')

cot = Function(name='cot',
               arguments=('z',),
               expression='1./cmath.tan(z)')
'''

INIT_TEMPLATE = '''\
from . import particles
from . import couplings
from . import lorentz
from . import parameters
from . import vertices
from . import coupling_orders
from . import function_library
from .object_library import all_particles, all_vertices, all_couplings, \\
    all_lorentz, all_parameters, all_orders, all_functions

gauge = [0]

__arxiv__ = ""
__author__ = "{author}"
__date__ = "{date}"
__version__ = "{version}"
'''
