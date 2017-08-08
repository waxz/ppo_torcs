#coding: utf-8

__all__ = ['logger', 'redict_logger_output']
from tensorpack.utils import logger
from .logmodule import getLogger

logger = getLogger()

def redict_logger_output(stream):
    for h in logger.handlers:
        h.stream = stream