"""Convenience functions built on top of `make_vjp`."""
from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
from inspect import getargspec
import warnings

from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace

import autograd.numpy as np

from autograd.builtins import list as ag_list, tuple as ag_tuple
from scipy.special import binom

make_vjp = unary_to_nary(_make_vjp)
make_jvp = unary_to_nary(_make_jvp)

@unary_to_nary
def grad(fun, x):
    """
    Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""
    vjp, ans = _make_vjp(fun, x)
    if not vspace(ans).size == 1:
        raise TypeError("Grad only applies to real scalar-output functions. "
                        "Try jacobian, elementwise_grad or holomorphic_grad.")
    return vjp(vspace(ans).ones())

@unary_to_nary
def elementwise_grad(fun, x):
    """
    Returns a function that computes the sum of each column of the Jacobian of
    `fun`, in one pass. If the Jacobian is diagonal, then this is the diagonal
    of the Jacobian.
    """
    vjp, ans = _make_vjp(fun, x)
    if vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")
    return vjp(vspace(ans).ones())

@unary_to_nary
def deriv(fun, x):
    return _make_jvp(fun, x)(vspace(x).ones())[1]

@unary_to_nary
def jacobian(fun, x):
    """
    Returns a function which computes the Jacobian of `fun` with respect to
    positional argument number `argnum`, which must be a scalar or array. Unlike
    `grad` it is not restricted to scalar-output functions, but also it cannot
    take derivatives with respect to some argument types (like lists or dicts).
    If the input to `fun` has shape (in1, in2, ...) and the output has shape
    (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
    """
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return np.reshape(np.stack(grads), jacobian_shape)

@unary_to_nary
def holomorphic_grad(fun, x):
    if not vspace(x).iscomplex:
        warnings.warn("Input to holomorphic_grad is not complex")
    return grad(lambda x: np.real(fun(x)))(x)

def grad_named(fun, argname):
    '''Takes gradients with respect to a named argument.
       Doesn't work on *args or **kwargs.'''
    arg_index = getargspec(fun).args.index(argname)
    return grad(fun, arg_index)

@unary_to_nary
def hessian(fun, x):
    "Returns a function that computes the exact Hessian."
    return jacobian(jacobian(fun))(x)

@unary_to_nary
def make_hvp(fun, x):
    """Builds a function for evaluating the Hessian-vector product at a point,
    which may be useful when evaluating many Hessian-vector products at the same
    point while caching the results of the forward pass."""
    return _make_vjp(grad(fun), x)

def hessian_tensor_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-tensor product.
    The returned function has arguments (*args, tensor, **kwargs), and for
    vectors takes roughly 4x as long to evaluate as the original function."""
    fun_grad = grad(fun, argnum)
    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(fun_grad(*args, **kwargs), vector, np.ndim(vector))
    return grad(vector_dot_grad, argnum)
hessian_vector_product = hessian_tensor_product

def tensor_jacobian_product(fun, argnum=0):
    """Builds a function that returns the exact tensor-Jacobian product, that
    is the Jacobian matrix left-multiplied by tensor. The returned function
    has arguments (*args, tensor, **kwargs)."""
    def vector_dot_fun(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(vector, fun(*args, **kwargs), axes=np.ndim(vector))
    return jacobian(vector_dot_fun, argnum)
vector_jacobian_product = tensor_jacobian_product

@unary_to_nary
def make_jvp_reversemode(fun, x):
    """Builds a function for evaluating the Jacobian-vector product at a
    point. Roughly 1.5x more FLOPs than forward-mode, plus memory requirements
    that scale with the number of primitives applied in the evaluation of f, as
    well as other overheads. See j-towns.github.io/2017/06/12/A-new-trick.html."""
    vjp, y = _make_vjp(fun, x)
    vjp_vjp, _ = _make_vjp(vjp, vspace(y).zeros())
    return vjp_vjp  # vjp_vjp is just jvp by linearity

# TODO(mattjj): update this function using make_jvp and const_graph
def make_ggnvp(f, g=lambda x: 1./2*np.sum(x**2, axis=-1), f_argnum=0):
    """Builds a function for evaluating generalized-Gauss-Newton-vector products
    at a point. Slightly more expensive than mixed-mode."""
    @unary_to_nary
    def _make_ggnvp(f, x):
        f_vjp, f_x = _make_vjp(f, x)
        g_hvp, grad_g_x = _make_vjp(grad(g), f_x)
        f_jvp, _ = _make_vjp(f_vjp, vspace(grad_g_x).zeros())
        def ggnvp(v): return f_vjp(g_hvp(f_jvp(v)))
        return ggnvp
    return _make_ggnvp(f, f_argnum)

@unary_to_nary
def value_and_grad(fun, x):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    vjp, ans = _make_vjp(fun, x)
    if not vspace(ans).size == 1:
        raise TypeError("value_and_grad only applies to real scalar-output "
                        "functions. Try jacobian, elementwise_grad or "
                        "holomorphic_grad.")
    return ans, vjp(vspace(ans).ones())

@unary_to_nary
def grad_and_aux(fun, x):
    """Builds a function that returns the gradient of the first output and the
    (unmodified) second output of a function that returns two outputs."""
    vjp, (ans, aux) = _make_vjp(lambda x: atuple(fun(x)), x)
    return vjp((vspace(ans).ones(), vspace(aux).zeros())), aux

def multigrad_dict(fun):
    "Takes gradients wrt all arguments simultaneously,"
    "returns a dict mapping 'argname' to 'gradval'"

    import funcsigs
    sig = funcsigs.signature(fun)

    def select(preds, lst):
        idx = lambda item: next(
            (i for i, pred in enumerate(preds) if pred(item)), len(preds))
        results = [[] for _ in preds] + [[]]
        for item in lst:
            results[idx(item)].append(item)
        return results

    is_var_pos = lambda name: sig.parameters[name].kind == sig.parameters[name].VAR_POSITIONAL
    is_var_kwd = lambda name: sig.parameters[name].kind == sig.parameters[name].VAR_KEYWORD
    var_pos, var_kwd, argnames = select([is_var_pos, is_var_kwd], sig.parameters)

    todict = lambda dct: {key:dct[key] for key in dct}

    def apply_defaults(arguments):
        defaults = {name: param.default for name, param in sig.parameters.items()
                    if param.default is not param.empty}
        return OrderedDict((name, arguments[name] if name in arguments else defaults[name])
                           for name in sig.parameters)

    def gradfun(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)

        args = lambda dct: tuple(dct[var_pos[0]]) if var_pos else ()
        kwargs = lambda dct: todict(dct[var_kwd[0]]) if var_kwd else {}
        others = lambda dct: tuple(dct[argname] for argname in argnames
                                   if argname not in var_kwd + var_pos)

        newfun = lambda dct: fun(*(others(dct) + args(dct)), **kwargs(dct))

        argdict = apply_defaults(bindings.arguments)
        grad_dict = grad(newfun)(dict(argdict))
        return OrderedDict((argname, grad_dict[argname]) for argname in argdict)

    return gradfun

def checkpoint(fun):
    """Returns a checkpointed version of `fun`, where intermediate values
    computed during the forward pass of `fun` are discarded and then recomputed
    for the backward pass. Useful to save memory, effectively trading off time
    and memory. See e.g. arxiv.org/abs/1604.06174.
    """
    def wrapped_grad(argnum, ans, args, kwargs):
        return make_vjp(fun, argnum)(*args, **kwargs)[0]
    wrapped = primitive(fun)
    defvjp_argnum(wrapped, wrapped_grad)
    return wrapped

def mut_loop_primitive(function, parameters, initial_condition, inputs):
    """Repeatedly applies function starting at initial_condition without 
    recording intermediate results
    """
    condition = initial_condition
    for input in inputs:
        condition = function(parameters, condition, input)
    return condition

# thanks to https://github.com/jrmaddison/tlm_adjoint/blob/master/python/tlm_adjoint/binomial_checkpointing.py for this code
def checkpoint_policy(n, snapshots):
    if n < 1:
        raise ValueError("Require at least one block")
    if snapshots <= 0:
        raise ValueError("Require at least one snapshot")
  
  # Discard excess snapshots
    snapshots = max(min(snapshots, n - 1), 1)  
    # Handle limiting cases
    if snapshots == 1:
        return n - 1  # Minimal storage
    elif snapshots == n - 1:
        return 1  # Maximal storage
  
    t = 2
    b_s_tm2 = 1
    b_s_tm1 = snapshots + 1
    b_s_t = ((snapshots + 1) * (snapshots + 2)) // 2
    while b_s_tm1 >= n or n > b_s_t:
        t += 1
        b_s_tm2, b_s_tm1, b_s_t = b_s_tm1, b_s_t, (b_s_t * (snapshots + t)) // t
  
  # Return the maximal step size compatible with Fig. 4 of GW2000
    b_sm1_tm2 = (b_s_tm2 * snapshots) // (snapshots + t - 2)
    if n <= b_s_tm1 + b_sm1_tm2:
        return n - b_s_tm1 + b_s_tm2
    b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
    b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
    if n <= b_s_tm1 + b_sm2_tm1:
        return b_s_tm2 + b_sm1_tm2
    elif n <= b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
        return n - b_sm1_tm1 - b_sm2_tm1
    else:
        return  b_s_tm1

# @profile
def make_bc_vjpmaker(function, sequence_length, num_checkpoints):
    assert(num_checkpoints >= 1)
    assert(sequence_length > 0)

    # @profile
    def vjpmaker(argnum, ans, args, kwargs):
        parameters, initial_state, inputs = args
        assert(sequence_length == len(inputs))

        # @profile
        def vjp_one_checkpoint(parameters, initial_state, inputs, visible_state_grads, hidden_state_grad):
            assert(len(inputs) > 0)
            assert(len(visible_state_grads) > 0)
            assert(len(inputs) + 1 == len(visible_state_grads))

            l = list(range(len(inputs)))
            l.reverse()

            visible_state_grad_vspace = vspace(visible_state_grads[0])
            hidden_state_grad_vspace = vspace(initial_state[0])

            parameter_vspace = vspace(parameters)
            parameter_grad = parameter_vspace.zeros()

            if hidden_state_grad is None:
                hidden_state_grad = hidden_state_grad_vspace.zeros()
                
            hidden_state_grads = [hidden_state_grad_vspace.zeros()] * len(inputs) + [hidden_state_grad]

            curried_function = lambda pc, input: function(pc[0], pc[1], input)
            curried_vjp = make_vjp(curried_function, 0)
            for y in l:
                state_y = mut_loop_primitive(function, parameters, initial_state, inputs[:y])
                
                ingrad_vjp = curried_vjp(ag_tuple((parameters, state_y)), inputs[y])[0]
                parameter_ingrad, (hidden_state_ingrad, visible_state_ingrad) = ingrad_vjp(
                    ag_tuple((hidden_state_grads[y + 1], visible_state_grads[y + 1]))
                )
                
                parameter_grad = parameter_vspace.add(parameter_grad, parameter_ingrad)
                hidden_state_grads[y] = hidden_state_grad_vspace.add(hidden_state_grads[y], hidden_state_ingrad)
                visible_state_grads[y] = visible_state_grad_vspace.add(visible_state_grads[y], visible_state_ingrad)

            return parameter_grad, ag_tuple((hidden_state_grads[0], visible_state_grads[0]))
        
        # @profile
        def vjp_general(parameters, initial_state, inputs, visible_state_grads, hidden_state_grad, num_checkpoints):
            assert(len(inputs) > 0)
            assert(len(visible_state_grads) > 0)
            assert(len(inputs) + 1 == len(visible_state_grads))

            if num_checkpoints == 1 or len(inputs) == 1:
                return vjp_one_checkpoint(parameters, initial_state, inputs, visible_state_grads, hidden_state_grad)
            else:
                y = checkpoint_policy(len(inputs), num_checkpoints)

                state_y = mut_loop_primitive(function, parameters, initial_state, inputs[:y])
                
                theta_ingrad, (hidden_state_ingrad, visible_state_ingrad) = vjp_general(
                    parameters, state_y, inputs[y:], visible_state_grads[y:], hidden_state_grad, num_checkpoints - 1
                )
                visible_state_grads[y] = visible_state_ingrad
                
                theta0_ingrad, state0_ingrad = vjp_general(
                    parameters, initial_state, inputs[:y], visible_state_grads[:y + 1], hidden_state_ingrad, num_checkpoints
                )
                
                return vspace(theta_ingrad).add(theta_ingrad, theta0_ingrad), state0_ingrad
        return lambda visible_state_grads: vjp_general(parameters, initial_state, inputs, visible_state_grads, None, num_checkpoints)[argnum]
    return vjpmaker
        
# @profile
def binomial_checkpoint(function, sequence_length, num_checkpoints):
    """
    Args:
        function: takes parameters, condition, and input and produces output. 
        sequence_length: sequence length
        num_checkpoints: number of checkpoints we can save

    Returns:
        wrapped: a new primitive whose gradient, when called, performs the checkpointing algorithm of Gruslys et. al (2016)
    """
    def loop_primitive(parameters, initial_state, inputs):
        state = initial_state
        visible_states = ag_list([state[1]])

        for i, input in enumerate(inputs):
            state = function(parameters, state, input)
            visible_states += ag_list([state[1]])
        return visible_states

    wrapped_grad = make_bc_vjpmaker(function, sequence_length, num_checkpoints)
    wrapped = primitive(loop_primitive)
    defvjp_argnum(wrapped, wrapped_grad)
    return wrapped




