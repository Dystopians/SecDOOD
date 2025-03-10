import mmcv
from mmcv import digit_version

from .version import __version__

mmcv_minimum_version = '1.2.6'
mmcv_maximum_version = '1.3'
mmcv_version = digit_version(mmcv.__version__)


__all__ = ['__version__']
