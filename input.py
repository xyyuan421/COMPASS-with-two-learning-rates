# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:22:01 2024

@author: xinyi
"""

import os, sys
if __name__ == '__main__':
    input_par = sys.argv[1:]
    type = input_par[0]
    criterion = input_par[1]
    if type == '1':
        os.system('python PowerAnalysis1.py {}'.format(criterion))
    if type == '2':
        os.system('python PowerAnalysis2.py {}'.format(criterion))