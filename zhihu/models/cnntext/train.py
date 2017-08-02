import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score

log = Tools.get_logger('cnn text')


