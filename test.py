from model import Model

import tensorflow as tf
import numpy as np
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

def main():
    ### create model
    model = Model()
    model.create_model()
    model.test()

if __name__ == '__main__':
    main()