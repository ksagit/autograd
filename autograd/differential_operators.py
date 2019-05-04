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

def forward_loop_no_saving(function, parameters, initial_condition, inputs):
    """Repeatedly applies function starting at initial_condition without 
    recording intermediate results
    """
    condition = initial_condition
    for input in inputs:
        condition = function(parameters, condition, input)
    return condition

profile = lambda fun: fun

@profile
def checkpoint_policy(sequence_length, num_checkpoints):
    def binomial_loss(y):
        return np.abs(binom(num_checkpoints * sequence_length / y, num_checkpoints) - sequence_length)
    if sequence_length < 2:
        raise ValueError("Invalid sequence length")
    else:
        if num_checkpoints < 1:
            raise ValueError("Invalid number of checkpoints")
        else:
            return sequence_length - np.argmin([binomial_loss(y + 1) for y in range(sequence_length)])

@profile
def make_bc_vjpmaker(function, sequence_length, num_checkpoints, postprocess):
    assert(num_checkpoints >= 1)
    assert(sequence_length > 0)

    @profile
    def vjpmaker(argnum, ans, args, kwargs):
        parameters, initial_state, inputs = args
        assert(sequence_length == len(inputs))

        curried_function = lambda param_and_state, input: function(param_and_state[0], param_and_state[1], input)
        curried_function_vjp = make_vjp(curried_function, 0)

        curried_postprocess = lambda param_and_vis_state: postprocess(param_and_vis_state[0], param_and_vis_state[1])
        curried_postprocess_vjp = make_vjp(curried_postprocess, 0)

        @profile
        def vjp_one_checkpoint(parameters, initial_state, inputs, postprocess_grads, hidden_state_grad_wrt_next_state, visible_state_grad_wrt_next_state):
            assert(len(inputs) > 0)
            assert(len(output_grads) > 0)
            assert(len(inputs) + 1 == len(output_grads))

            hidden_state_grad_vspace = vspace(initial_state[0])
            visible_state_grad_vspace = vspace(initial_state[1])
            parameter_vspace = vspace(parameters)

            parameter_grad = parameter_vspace.zeros()

            if hidden_state_grad_wrt_next_state is None:
                hidden_state_grad_wrt_next_state = hidden_state_grad_vspace.zeros()
            if visible_state_grad_wrt_next_state is None:
                visible_state_grad_wrt_next_state = visible_state_grad_vspace.zeros()
                
            for y in range(len(inputs) - 1, -1, -1):
                state_y = forward_loop_no_saving(function, parameters, initial_state, inputs[:y])
                
                state_vjp, (_, visible_state_yplusone) = curried_function_vjp(ag_tuple((parameters, state_y)), inputs[y])
                postprocess_vjp = curried_postprocess_vjp((parameters, visible_state_yplusone))[0]

                parameter_grad_wrt_output, visible_state_grad_wrt_output = postprocess_vjp(postprocess_grads[y])
                parameter_grad_wrt_next_state, (hidden_state_grad_wrt_next_state, visible_state_grad_wrt_next_state) = state_vjp(
                    ag_tuple((
                            hidden_state_grad_wrt_next_state, 
                            visible_state_grad_vspace.add(visible_state_grad_wrt_output, visible_state_grad_wrt_next_state)
                        )
                    )
                )
                parameter_grad = parameter_vspace.add(parameter_grad, parameter_vspace.add(parameter_grad_wrt_output, parameter_grad_wrt_next_state))

            return parameter_grad, ag_tuple(hidden_state_grad_wrt_next_state, visible_state_grad_wrt_next_state)
        
        @profile
        def vjp_general(parameters, initial_state, inputs, visible_state_grads, hidden_state_grad, num_checkpoints):
            assert(len(inputs) > 0)
            assert(len(visible_state_grads) > 0)
            assert(len(inputs) + 1 == len(visible_state_grads))

            if num_checkpoints == 1 or len(inputs) == 1:
                return vjp_one_checkpoint(parameters, initial_state, inputs, visible_state_grads, hidden_state_grad)
            else:
                y = checkpoint_policy(len(inputs), num_checkpoints)

                state_y = forward_loop_no_saving(function, parameters, initial_state, inputs[:y])
                
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
        
@profile
def binomial_checkpoint(function, sequence_length, num_checkpoints, postprocess=lambda p, x: x):
    """
    Args:
        function: takes parameters, (hidden_state, visible_state), and input and produces (hidden_state, visible_state) 
        sequence_length: sequence length
        num_checkpoints: number of checkpoints we can save

    Returns:
        wrapped: a new primitive whose gradient, when called, performs the checkpointing algorithm of Gruslys et. al (2016)
    """

    """
    def loop_primitive_mut(parameters, initial_state, inputs):
        mutstate = {0: initial_state}
        def mut_propagate(input):
            mutstate[0] = function(parameters, mutstate[0], input)
            return mutstate[0][1]

        return [initial_state[1]] + [mut_propagate(input) for input in inputs]
    """
    def loop_primitive(parameters, initial_state, inputs):
        state = initial_state
        outputs = ag_list([postprocess(parameters, state[1])])

        for input in inputs:
             state = function(parameters, state, input)
             outputs += ag_list([postprocess(parameters, state[1])])

        return outputs
                
    wrapped_grad = make_bc_vjpmaker(function, sequence_length, num_checkpoints, postprocess)
    wrapped = primitive(loop_primitive)
    defvjp_argnum(wrapped, wrapped_grad)
    return wrapped




