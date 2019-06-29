#-*-coding:utf-8-*-
from scipy.stats import pearsonr
import numpy as np

def read_document(array_1, array_2):
	result = pearsonr(array_1, array_2)
	return result
