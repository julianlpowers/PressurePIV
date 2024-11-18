import numpy as np
import sympy as sp
import inspect


def to_symbolic_expression(f, syms_string=None):
    non_trivial_replacements = { #numpy diction -> sympy diction
        'arcsin': 'asin',  
        'arccos': 'acos',  
        'arctan': 'atan',  
        'arctan2': 'atan2',  
        'arcsinh': 'asinh',  
        'arccosh': 'acosh',  
        'arctanh': 'atanh', 
        'abs': 'Abs',  
        'ceil': 'ceiling',  
        'conj': 'conjugate',  
        'real': 're',  
        'imag': 'im',  
        # Add more replacements as needed
    }

    source = inspect.getsource(f)
    source = source.replace(source.split('(')[0],'def g')
    for np_func, sp_func in non_trivial_replacements.items():
            source = source.replace(np_func, sp_func)
    source = source.replace('np.','sp.')
    
    ldict = {}
    exec(source,globals(),ldict)

    g = ldict['g']

    if syms_string==None:
        syms_string = source.split(')')[0].split('(')[1]
        
    syms = sp.symbols(syms_string)
    
    if len(syms) > 1:
        return g(*syms), syms
    else:
        return g(syms), syms


def diff(f, diff_key, syms_string=None, verbose=False):
    expr,syms = to_symbolic_expression(f,syms_string)
    
    sym_dict = {}
    for sym in syms:
        sym_dict[sym.name] = sym
        
    for char in diff_key:
        sym = sym_dict[char]
        expr = sp.diff(expr,sym)
        
    if verbose:
        print('D_' + diff_key + '(f)','->',expr)

    return sp.lambdify(syms,expr,'numpy')
    
        
    

if __name__ == '__main__':

    def f(x, y, z):
        return x**2#np.sin(x) * np.exp(y) + np.log(x + y) + z


    fx = diff(f,'y')

    print(fx(1,1,1))
        
    






    
    
