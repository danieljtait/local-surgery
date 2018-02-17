import scipy.special
import json
import itertools, collections
import sympy
from sympy.parsing import sympy_parser
from gpode.gaussianprocesses import MultioutputGaussianProcess
from gpode.kernels import MultioutputKernel
import os


class NestedIntegralGaussianProcess(MultioutputGaussianProcess):
    pass

####
#
#
# Methods relating to loading covariance and mean functions from
# stored sympy string resources
#
#
#

def lambda_from_sympystr(str_expr, symb_names, print_it=False):
    sympy_expr = sympy_parser.parse_expr(str_expr)
    free_symbols = [symb for symb in sympy_expr.free_symbols]

    # Wish to collect the... order important
    symb_map = collections.OrderedDict()
    remaining_names = []  # symbols not found, but order still important
    for name in symb_names:
        name_found = False
        for symb in free_symbols:
            if name == str(symb):
                symb_map[name] = symb
                free_symbols.remove(symb)
                name_found = True
                break
        if not name_found:
            symb_map[name] = sympy.Symbol(name)
            #remaining_names.append(name)

    fun_args = tuple(item for item in
                     itertools.chain(symb_map.values(), remaining_names))
    lfunc = sympy.lambdify(fun_args,
                           sympy_expr,
                           modules=["numpy",
                                    {"erf": scipy.special.erf}])
    return lfunc


###
# Generic class for creating lambda functions
# from stored sympy string expressions
class SympyStrToLambdaLoader:

    @classmethod
    def load(cls, config):
        raise NotImplementedError


##
# ToDo: Load expectations
class NestedGPIntegralExpectLoader(SympyStrToLambdaLoader):

    @classmethod
    def load(cls, config):
        raise NotImplementedError


##
#
class NestedGPIntegralCovarLoader(SympyStrToLambdaLoader):

    @classmethod
    def load(cls, config):
        fname = config["fname"]

        dirname = os.path.split(__file__)[0]
        fname = os.path.join(dirname, fname)

        snames = config["symb_names"]

        try:
            with open(fname, 'r') as f:
                data = f.read()
            obj = json.loads(data)
            for item in obj.items():
                lf = lambda_from_sympystr(item[1], snames)
                obj[item[0]] = lf
            return obj
        except:
            # ToDo
            # Raise appropriate error for
            # i) file not existing
            # ii) required symbols not found
            # iii) sympy parser failer etc.
            pass


def ngpintegrals_sqexpcovs():
    config = {"fname": "myfile2.txt",
              "symb_names": ["s", "t", "s_0", "t_0", "theta_0", "theta_1"]}
    return NestedGPIntegralCovarLoader.load(config)
