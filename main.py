"""
Standard Simplex Induced Clustering with
Hierarchical Deep Dictionary Learning

Application : FASHION MNIST
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
----------------------------------------------------------
------            Main application file              -----


"""

import numpy as np
import os,sys
import argparse
import tensorflow as tf
from model import ConvSSHDDLC

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# --Network architecture parameters ------------
# ----------------------------------------------
ap.add_argument("-NCONV_I", "--NCONV_I", required=True, help="Number of input convolutional layers.")
ap.add_argument("-NDL",   "--NDL", required=True, help="Number of dictionary learning layers.")
ap.add_argument("-NC",    "--NC", required=True, help="Number of clustering layers.")
ap.add_argument("-nch",    "--nch", required=True, help="Number of color channels.")
# --Convolutional layers parameters ------------
ap.add_argument("-N_FILT_I","--N_FILT_I", required=True, help="Comma separated number of filters per input convolutional layer.")
ap.add_argument("-H_FILT_I","--H_FILT_I", required=True, help="Comma separated kernel size for filters at each input convolutional layer.")
ap.add_argument("-MP_I", "--MP_I", required=True, help="Comma separated binary values indicating maxpool (1) or not (0) after each input convolutional layer respectively.")
ap.add_argument("-D_I", "--D_I", required=True, help="Comma separated binary values indicating dropout layer (1) or not (0) after each input convolutional layer respectively.")
ap.add_argument("-drop", "--dropout", required=True, help="Dropout parameter (between (0,1)).")
# --Dict. Learning layers parameters -----------
ap.add_argument("-l_l1", "--lambda_l1", required = True, help = "Lambda L1 penalization parameter for all dictionary layers.")
ap.add_argument("-K","--K", required=True, help="Number of ISTA-like iterations (nodes) for each dictionary learning layer." )
ap.add_argument("-n_atoms", "--number_atoms", required = True, help = "Comma separated number of atoms per dictionary layer.")
ap.add_argument("-n_dicts", "--number_dictionaries", required = True, help = "Comma separated number of dictionaries per HDDL layer.")
# --Clustering layers parameters ---------------
ap.add_argument("-out_sc1","-out_sc1", required=True, help= "Comma separated numer of nodes per clustering layer.")
# --Optimization parameters --------------------
ap.add_argument("-l_r","--learning_rate", required=True, help="Learning rate for optimization algorithm.")
ap.add_argument("-n_e","--number_epochs", required=True, help="Number of epochs for training stage.")
ap.add_argument("-b_s","--batch_size", required=True, help="Batch size for training stage.")
ap.add_argument("-d_s","--display_step", required=True, help="How often to display training information (every display_step times).")
# --Other parameters ---------------------------
# ----------------------------------------------
ap.add_argument("-in_f", "--input_folder", required = True, help="Input folder.")
ap.add_argument("-ou_f", "--output_folder", required = True, help="Output folder.")
ap.add_argument("-n",    "--n", required=True, help="Number of total samples.")
ap.add_argument("-mu",  "--mu", required=True, help="Weight for the expected probability term in the loss function.")
ap.add_argument("-nrows","--nrows", required=True, help="Number of rows (for images).")
ap.add_argument("-ncols","--ncols", required=True, help="Number of columns (for images).")
ap.add_argument("-NCLUST","--NCLUST", required=True, help="Number of clusters.")
args = vars(ap.parse_args())

NConvI           = int(args['NCONV_I'])
NDL              = int(args['NDL'])
NC               = int(args['NC'])
nch              = int(args['nch'])
nFiltI_str       = str(args['N_FILT_I']).split(',')
nFiltI           = []
for i in range(0,NConvI):
   nFiltI.append(int(nFiltI_str[i]))
hFiltI_str       = str(args['H_FILT_I']).split(',')
hI               = []
for i in range(0,NConvI):
   hI.append( int(hFiltI_str[i]) )
mPoolI_str       = str(args['MP_I']).split(',')
mPoolI           = []
for i in range(0,NConvI):
   mPoolI.append( int(mPoolI_str[i])  )
dropI_str        = str(args['D_I']).split(',')
dropI            = []
for i in range(0,NConvI):
   dropI.append( int(dropI_str[i])  )
drop             = float(args['dropout'])


LambdaL1         = float(args['lambda_l1'])
K                = int(args['K'])
nAtoms_str       = str(args['number_atoms']).split(',')
nDicts_str       = str(args['number_dictionaries']).split(',')
nAtoms           = []
nDicts           = []
for i in range(0,NDL):
   nAtoms.append( int(nAtoms_str[i]) )
   nDicts.append( int(nDicts_str[i]) )
out_sc1_str      = str(args['out_sc1']).split(',')
out_sc1          = []
for i in range(0,NC):
   out_sc1.append( int(out_sc1_str[i]) )
learning_rate    = float(args['learning_rate'])
num_epochs       = int(args['number_epochs'])
batch_size       = int(args['batch_size'])
display_step     = int(args['display_step'])
input_folder     = str(args['input_folder'])
output_folder    = str(args['output_folder'])
n                = int(args['n'])
mu               = float(args['mu'])
nrows            = int(args['nrows'])
ncols            = int(args['ncols'])
NCLUST           = int(args['NCLUST'])
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main(_):
  # uncomment the next line if problems with version 1 compatibility
  # and eager execution
  # --------------------------------------------------------------------
  tf.compat.v1.disable_eager_execution()
  # --------------------------------------------------------------------
  with tf.compat.v1.Session() as sess:
     conv_sshddlc = ConvSSHDDLC(sess, \
                      NConvI, NDL, NC, nch,\
                      nFiltI,hI,mPoolI,dropI,drop, LambdaL1,K,nAtoms,nDicts, out_sc1,\
                      learning_rate,num_epochs,batch_size,display_step,\
                      input_folder,output_folder,\
                      nrows,ncols, n, mu,NCLUST)
     conv_sshddlc.train()

if __name__ == '__main__':
  tf.compat.v1.app.run()

