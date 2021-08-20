# -*- coding: utf-8 -*-
from .stoch import UniVar, MultiVar
from duqo.optimization.rrdo import RRDO
from duqo.optimization.space import InputSpace, FullSpace
from duqo.optimization.predict import CondMom, CondProba
ConditionalMoment = CondMom  # So  that both are valid
ConditionalProbability = CondProba
