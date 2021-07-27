# -*- coding: utf-8 -*-
from .stoch import UniVar, MultiVar
from .rrdo import RRDO 
from pyRDO.optimization.space import InputSpace, FullSpace
from pyRDO.optimization.predict import CondMom, CondProba
ConditionalMoment = CondMom
ConditionalProbability = CondProba