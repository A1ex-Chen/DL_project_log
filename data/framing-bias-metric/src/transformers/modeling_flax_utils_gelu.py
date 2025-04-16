@jax.jit
def gelu(x):
    """
    Gaussian error linear unit activation function.

    Computes the element-wise function:

    .. math::
      \\mathrm{gelu}(x) = \\frac{x}{2} \\left(1 + \\mathrm{tanh} \\left(
        \\sqrt{\\frac{2}{\\pi}} \\left(x + 0.044715 x^3 \\right) \\right) \\right)

    We explicitly use the approximation rather than the exact formulation for speed. For more information, see
    `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_, section 2.
    """
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
