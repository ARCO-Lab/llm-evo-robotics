import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from map_elites_core import Individual, RobotGenotype, RobotPhenotype, FeatureExtractor