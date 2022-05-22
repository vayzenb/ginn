import warnings
warnings.filterwarnings("ignore")
import resource
import sys
import time
import os
import gc
import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

from nilearn import image, datasets
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball