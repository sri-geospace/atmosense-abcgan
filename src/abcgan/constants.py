"""
File for global constants used in the program. Import for defining the structure and number of variables in the driver parameters, number of altitude bins, output data products, ...

"""
import numpy as np

# ML parameters
batch_size = 100

# Feature sizes
max_alt = 30

# Alt bins for metrics
alt_bins = [[0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18],
            [19, 20, 22, 23, 24],
            [25, 26, 27, 28, 29]]

# Alt bins for metrics
alt_bins_all = [[0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [17],
            [18],
            [19],
            [20],
            [21],
            [22],
            [23],
            [24],
            [25],
            [26],
            [27],
            [28],
            [29]]

# List types of background variables
bv_vars = ['Ne', 'Te', 'Ti']
mean_name = 'bac'
std_name = 'rms'
bv_names = [v + '_' + s
            for v in bv_vars
            for s in (mean_name, std_name)]

n_bv = len(bv_names)

# list types of drivers
driver_names = ['Ap', 'F10.7', 'F10.7avg',
                'MLT', 'SLT', 'SZA', 'ap']

# driving parameters that are cyclic, value is assumed equal to 0
cyclic_driver = {
    'MLT': 24.0,
}

n_driver = len(driver_names)

driver_feat_names = []
for dn in driver_names:
    if dn in cyclic_driver:
        driver_feat_names.append('cos_' + dn)
        driver_feat_names.append('sin_' + dn)
    else:
        driver_feat_names.append(dn)

# temporarily have as many features as drivers/bvs
n_driver_feat = len(driver_feat_names)
n_bv_feat = n_bv

bv_mu = np.array(
    [25.019377, 23.48621, 7.1187057,
     5.4906588, 6.9509993, 5.604995])
bv_sigma = np.array(
    [1.0008324, 0.986352, 0.8379408,
     1.316161, 0.7411894, 1.3654965])
driver_mu = np.array(
    [1.91605153, 4.55362418, 4.55630713, -0.82761988,
     -0.69772302, 2.37613748, 4.43544436, 1.79978106])
driver_sigma = np.array(
    [0.67652099, 0.26991646, 0.25331369, 1.82686466,
     1.88726553, 0.70034762, 0.27227722, 0.85919697])

bv_thresholds = np.array(
    [[-1, 288214929284788.94],
     [1, 1892646860000.0],
     [-1, 500000],
     [-1, 4395068574.6624],
     [-1, 100247.0],
     [-1, 8.454286361206116e+6]])

assert((n_bv_feat,) == bv_mu.shape)
assert((n_bv_feat,) == bv_sigma.shape)
assert((n_driver_feat,) == driver_mu.shape)
assert((n_driver_feat,) == driver_sigma.shape)
assert((n_bv, 2) == bv_thresholds.shape)
