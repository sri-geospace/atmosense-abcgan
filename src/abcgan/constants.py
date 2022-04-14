"""
File for global constants used in the program.
"""
import numpy as np

# Driver augmentation delay cond
dr_delay = 2 * 3600  # 2 hours

# ML parameters
batch_size = 100

# Feature sizes
max_alt = 30
max_alt_lidar = 20
max_n_waves = 3
n_waves = 1

# Alt bins for metrics
alt_bins = [[0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18],
            [19, 20, 22, 23, 24],
            [25, 26, 27, 28, 29]]

# Alt bins for metrics
alt_bins_lidar = [[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8, 9],
                  [10, 11, 12],
                  [14, 15, 16],
                  [17, 18, 19]]

altgrid = np.concatenate((np.arange(92., 119., 4.5),
                          np.arange(119., 146., 9.),
                          np.arange(146., 200., 18.),
                          np.arange(200., 800., 24.)))

# List of hfp variables
hfp_names = ['GW_tau', 'GW_lamh', 'GW_lamv', 'GW_ch', 'GW_cv', 'GW_phi', 'GW_hl', 'GW_hu']
hfp_units = ['min', 'km', 'km', 'm/s', 'm/s', 'degree', 'km', 'km']
n_hfp = len(hfp_names)

# HFP features to log scale
log_hfp = [0, 1, 2, 3, 4]
invert_hfp = [2, 4]

# z_scaled clip ranges for each hfp feats for hellinger dist calculation
hfp_z_ranges = [(-2.4, 4), (-4, 4), (-4, 4), (-4, 4), (-4, 4), (-1.5, 1.8),  (-0.7, 4), (-4, 1.5)]

# constants used to determine number of bins
# and averaging filter length to use during hellinger dist calculation
bin_exp = 55 / 100
filter_exp = 1 / 3



# List types of background variables
bv_vars = ['Ne', 'Te', 'Ti', 'Ve', 'Vn', 'Vu']
mean_name = 'bac'
std_name = 'rms'
bv_names = [v + '_' + s
            for v in bv_vars
            for s in (mean_name, std_name)]

n_bv = len(bv_names)

bv_full_names = [r'Electron Density', r'RMS Electron Density', r'Electron Temp',
                 r'RMS Electron Temp', r'Ion Temp', r'RMS Ion Temp',
                 r'Northern Vel', r'RMS Northern Vel',
                 r'Upper Vel', r'RMS Upper Vel',
                 r'Eastward Vel', r'RMS Eastward Vel', ]

bv_units = [r'Density (m$^{-3}$)', r'Density (m$^{-3}$)', r'Temp (k)',
            r'Temp (k)', r'Temp (k)', r'Temp (k)',
            r'Vel (m/s)', r'Vel (m/s)',
            r'Vel (m/s)', r'Vel (m/s)',
            r'Vel (m/s)', r'Vel (m/s)', ]

lidar_bv_names = ['Tn_bac', 'dTn_bac', 'Nn_bac', 'dNn_bac']
n_lidar_bv = len(lidar_bv_names)

# list types of drivers
driver_names = ['Ap', 'F10.7', 'F10.7avg',
                'MLT', 'SLT', 'SZA', 'ap',
                'MEI', 'RMM1', 'RMM2', 'TCI',
                'moon_phase', 'moon_x', 'moon_y', 'moon_z']

driver_names_full = [r'Daily Ap Index', r'F10.7 Solar Radio Flux', r'Avg F10.7 Solar Radio Flux',
                     r'Magnetic Local Time', r'Solar Local Time', r'Solar Zenith Angle',
                     r'ap Geomagnetic Index', f'Multivariate ESNO Index', r'First MJO Index',
                     r'Second MJO Index', r'Thermosphere Climate Index',
                     r'Moon Phase', f'Lunar x Position', f'Lunar y Position', f'Lunar z Position']

driver_units = [r'Index Value', r'Flux Value', r'Flux Value',
                r'Hour', r'Hour', r'Degree',
                r'Index Value', r'Index Value',
                r'Index Value', r'Index Value',
                r'Index Value', r'Degree', f'ECEF Position', f'ECEF Position', f'ECEF Position']

# driving parameters that are cyclic, value is assumed equal to 0
cyclic_driver = {
    'MLT': 24.0,
    'SLT': 24.0,
    'moon_phase': 360.0
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
n_hfp_feat = n_hfp
n_lidar_bv_feat = n_lidar_bv

dr_feat_map = {}
idx = 0
for name in driver_names:
    if name in [c for c in cyclic_driver]:
        dr_feat_map[f'{name}'] = [idx, idx + 1]
        idx += 2
    else:
        dr_feat_map[f'{name}'] = [idx]
        idx += 1
del idx

log_bvs = [0, 1, 2, 3, 4, 5, 7, 9, 11]
log_drs = [0, 1, 2, 7, 8, 12]

bv_mu = np.array(
    [25.019377, 23.48621, 7.1187057,
     5.4906588, 6.9509993, 5.604995,
     -10.593687, 4.289111, 11.087654,
     4.077132, 1.852790, 2.672844,
     ])
lidar_bv_mu = np.array([5.4422550, 1.10414981,
                        0.14230881, 0.000443083])
bv_sigma = np.array(
    [1.0008324, 0.986352, 0.8379408,
     1.316161, 0.7411894, 1.3654965,
     85.712807, 0.738520, 162.111908,
     0.679040, 33.339760, 0.617492,
     ])

lidar_bv_sigma = np.array([0.0819466, 0.6514002,
                           0.1760367, 0.0003038])
driver_mu = np.array(
    [1.94544, 4.56088, 4.56631, -0.05525,
     0.01809, -0.05894, 0.00226, 4.43704,
     1.82567, -0.23761, 0.02396, 0.00368,
     25.10573, -0.04403, 0.01957, -20615.59992,
     6914.52185, 1031.51040])
driver_sigma = np.array(
    [0.68243, 0.26786, 0.25001, 0.70496,
     0.70686, 0.69716, 0.71449, 0.27556,
     0.85765, 0.96971, 1.01594, 0.99785,
     0.56443, 0.70315, 0.70941, 270407.64476,
     273149.93455, 24525.017936])

bv_thresholds = np.array(
    [[-1, 288214929284788.94],
     [1, 1892646860000.0],
     [-1, 500000],
     [-1, 4395068574.6624],
     [-1, 100247.0],
     [-1, 8.454286361206116e+6],
     [-2000.0, 2000.0],
     [0.000001, 2000.0],
     [-2000.0, 2000.0],
     [0.000001, 2000.0],
     [-2000.0, 2000.0],
     [0.000001, 2000.0],
     ])
lidar_thresholds = np.array(
    [[100, 324.35693359375],
     [-1, 50.0],
     [-1, 0.9010207056999207],
     [-1, 0.0025]])


hfp_mu = np.array(
    [3.68874, 6.37635, 5.96192, 5.53278,
     5.12021, -16.48782, 183.13432, 285.08089])
hfp_sigma = np.array(
    [0.38866, 0.92366, 1.14979, 0.93427,
     1.12458, 119.12195, 13.99695, 30.87417])

hfp_thresholds = np.array(
    [[0, 250.0],
     [0, 8000.0],
     [-8000.0, 0.0],
     [0.0, 4000.0],
     [-4000.0, 0.0],
     [-180.0, 180.0],
     [175.0, 275.0],
     [180.0, 325.0],
     ])

assert ((n_bv_feat,) == bv_mu.shape)
assert ((n_bv_feat,) == bv_sigma.shape)
assert ((n_hfp_feat,) == hfp_mu.shape)
assert ((n_hfp_feat,) == hfp_sigma.shape)
assert ((n_lidar_bv_feat,) == lidar_bv_mu.shape)
assert ((n_lidar_bv_feat,) == lidar_bv_sigma.shape)
assert ((n_driver_feat,) == driver_mu.shape)
assert ((n_driver_feat,) == driver_sigma.shape)
assert ((n_bv, 2) == bv_thresholds.shape)
assert ((n_hfp, 2) == hfp_thresholds.shape)
assert ((n_lidar_bv, 2) == lidar_thresholds.shape)
