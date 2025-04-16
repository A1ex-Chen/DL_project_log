import operator
from functools import reduce


__all__ = ['handlers']


























handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::baddbmm', baddbmm),
    (('aten::linear', 'aten::matmul'), matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    (
        (
            'aten::batch_norm',
            'aten::instance_norm',
            'aten::layer_norm',
            'aten::group_norm',
        ),
        norm,
    ),
    (
        (
            'aten::adaptive_avg_pool1d',
            'aten::adaptive_avg_pool2d',
            'aten::adaptive_avg_pool3d',
            'aten::avg_pool1d',
            'aten::avg_pool2d',
            'aten::avg_pool3d',
            'aten::mean',
        ),
        avg_pool_or_mean,
    ),
    ('aten::leaky_relu', leaky_relu),
    ('aten::upsample_bilinear2d', upsample_bilinear2d),
    (
        (
            'aten::adaptive_max_pool1d',
            'aten::adaptive_max_pool2d',
            'aten::adaptive_max_pool3d',
            'aten::add',
            'aten::add_',
            'aten::alpha_dropout',
            'aten::cat',
            'aten::chunk',
            'aten::clamp',
            'aten::clone',
            'aten::constant_pad_nd',
            'aten::contiguous',
            'aten::detach',
            'aten::div',
            'aten::div_',
            'aten::dropout',
            'aten::dropout_',
            'aten::embedding',
            'aten::eq',
            'aten::feature_dropout',
            'aten::flatten',
            'aten::floor',
            'aten::floor_divide',
            'aten::gt',
            'aten::hardtanh_',
            'aten::hardtanh',
            'aten::index',
            'aten::int',
            'aten::log_softmax',
            'aten::lt',
            'aten::max_pool1d',
            'aten::max_pool1d_with_indices',
            'aten::max_pool2d',
            'aten::max_pool2d_with_indices',
            'aten::max_pool3d',
            'aten::max_pool3d_with_indices',
            'aten::max_unpool1d',
            'aten::max_unpool2d',
            'aten::max_unpool3d',
            'aten::ne',
            'aten::reflection_pad1d',
            'aten::reflection_pad2d',
            'aten::reflection_pad3d',
            'aten::relu',
            'aten::relu_',
            'aten::replication_pad1d',
            'aten::replication_pad2d',
            'aten::replication_pad3d',
            'aten::rsub',
            'aten::select',
            'aten::sigmoid',
            'aten::size',
            'aten::slice',
            'aten::softmax',
            'aten::softshrink',
            'aten::squeeze',
            'aten::stack',
            'aten::sub',
            'aten::sum',
            'aten::t',
            'aten::tanh',
            'aten::threshold',
            'aten::to',
            'aten::transpose',
            'aten::upsample_nearest2d',
            'aten::view',
            'aten::zeros',
            'prim::constant',
            'prim::listconstruct',
            'prim::listunpack',
            'prim::numtotensor',
            'prim::tupleconstruct',
        ),
        None,
    ),
)