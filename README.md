# CONV_SSHDDLC_Tensorflow
Tensorflow implementation of Standard Simplex Induced Clustering with Hierarchical Deep Dictionary Learning.

## Python libraries required
 * tensorflow (compatibility with version 1)
 * munkres (for mapping cluster labels to groundtruth labels) [https://pypi.org/project/munkres/]
 * numpy
 * sklearn (to compute clustering metrics)
 * pickle (to save results)
 * gzip (to read compressed Fashion files)

## Usage
To obtain results for Fashion MNIST: `python main.py -NCONV_I 3 -NDL 2 -NC 3 -nch 1 -N_FILT_I 40,64,64 -H_FILT_I 3,5,5 -MP_I 1,1,1 -D_I 1,1,1 -drop 0.4 -l_l1 1.0   -K 30 -n_atoms 1800,100 -n_dicts 1,10 -out_sc1 90,40,1 -l_r 0.0005 -n_e 70 -b_s 500 -d_s 1 -in_f FASHION_MNIST_DIRECTORY -ou_f OUTPUT_DIRECTORY -n 70000 -mu 1.0 -ncols 28 -nrows 28 -NCLUST 10`
<br>

