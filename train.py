from model import Model

import tensorflow as tf
import numpy as np
import pdb
import os
from hparams import *

os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

def main():
    ### create model
    model = Model()
    model.create_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()