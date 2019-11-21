# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:06:26 2019

@author: Kwoks
"""

import json

def dump_json(data, path):
    '''
    :param data: list/dict, data need to be dumped
    :param path: str, the dump path of the data
    we us four indent
    '''
    with open(path, mode='w') as file:
        json.dump(data, file, indent=4)
        
def load_json(path):
    '''
    :param path:str, data path to be loaded
    :return res: list/dict, loaded data
    '''
    with open(path, mode='r') as file:
        res = json.load(file)
    return res
