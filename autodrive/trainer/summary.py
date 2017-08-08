#coding: utf-8
import numpy as np
import pandas as pd

def addParamSummary(*summary_lists):
    """
    Add summary Ops for all trainable variables matching the regex.

    Args:
        summary_lists (list): each is (regex, [list of summary type to perform]).
        Summary type can be 'mean', 'scalar', 'histogram', 'sparsity', 'rms'
    """
    from tensorpack.tfutils.tower import get_current_tower_context
    from tensorpack.utils.develop import log_deprecated
    from tensorpack.tfutils.symbolic_functions import rms
    import re
    import tensorflow as tf
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    if len(summary_lists) == 1 and isinstance(summary_lists[0], list):
        log_deprecated(text="Use positional args to call add_param_summary() instead of a list.")
        summary_lists = summary_lists[0]

    def perform(var, action):
        ndim = var.get_shape().ndims
        name = var.name.replace(':0', '')
        if action == 'scalar':
            assert ndim == 0, "Scalar summary on high-dimension data. Maybe you want 'mean'?"
            tf.summary.scalar(name, var)
            return
        assert ndim > 0, "Cannot perform {} summary on scalar data".format(action)
        if action == 'histogram':
            tf.summary.histogram(name, var)
            return
        if action == 'sparsity':
            tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(var))
            return
        if action == 'mean':
            tf.summary.scalar(name + '-mean', tf.reduce_mean(var))
            return
        if action == 'rms':
            tf.summary.scalar(name + '-rms', rms(var))
            return
        if action == 'absmax':
            tf.summary.scalar(name + '-absmax', tf.reduce_max(tf.abs(var)))
            return
        raise RuntimeError("Unknown summary type: {}".format(action))

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.name_scope('00/SummaryParam'):
        for p in params:
            name = p.name
            for rgx, actions in summary_lists:
                if not rgx.endswith('$'):
                    rgx = rgx + '(:0)?$'
                if re.match(rgx, name):
                    for act in actions:
                        perform(p, act)
