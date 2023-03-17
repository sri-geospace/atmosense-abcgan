"""
File for global constants used in the program.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Radar BV Constants
# ----------------------------------------------------------------------------
bv_vars = ['Ne', 'Te', 'Ti', 'Ve', 'Vn', 'Vu']
bv_vars_full = [r'$e^-\/\mathrm{Density}$', r'$e^-\/\mathrm{Temp}$', r'$\mathrm{Ion\/Temp}$',
                r'$\mathrm{North\/V}$', r'$\mathrm{Upper\/V}$', r'$\mathrm{East\/V}$']
bv_names = [v + '_' + s for v in bv_vars for s in ['bac', 'rms']]
bv_full_names = [v[:-1] + r'_\mathrm{' + s + '}$' for v in bv_vars_full for s in ['avg', 'rms']]
bv_units = [v for v in ['$m^3$', r'$k$', r'$k$', r'$m/s$', r'$m/s$', r'$m/s$'] for s in bv_vars]
n_bv = len(bv_names)
max_alt = 30  # Default Max Altitude Bins for BV Profiles

log_bvs = [0, 1, 2, 3, 4, 5, 7, 9, 11]  # BV to log scale
bv_z_ranges = [(-3, 3), (-3, 3), (-3, 2), (-3, 2.5), (-3, 2), (-4, 2),
               (-4, 4), (-3, 3), (-3, 3), (-3, 3), (-3, 3), (-4, 3.5)]
bv_meas_ranges = [(3.64593906e+09, 1.47824132e+12), (8.21920307e+08, 3.05515393e+11),
                  (9.89713155e+01, 6.59743184e+03), (3.67475054e+00, 6.50878234e+03),
                  (1.12004891e+02, 4.59700641e+03), (1.53740185e-01, 4.17032079e+03),
                  (-3.53444915e+02, 3.32257541e+02), (6.95301419e+00, 6.67255777e+02),
                  (-4.75248070e+02, 4.97423378e+02), (6.69070149e+00, 4.51257632e+02),
                  (-9.81664900e+01, 1.01872070e+02), (2.24920569e-01, 1.24721105e+02)]

bv_mu = np.array(
    [25.019377, 23.48621, 7.1187057,
     5.4906588, 6.9509993, 5.604995,
     -10.593687, 4.289111, 11.087654,
     4.077132, 1.852790, 2.672844,
     ])
bv_sigma = np.array(
    [1.0008324, 0.986352, 0.8379408,
     1.316161, 0.7411894, 1.3654965,
     85.712807, 0.738520, 162.111908,
     0.679040, 33.339760, 0.617492,
     ])
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

altgrid = np.concatenate((np.arange(92., 119., 4.5),
                          np.arange(119., 146., 9.),
                          np.arange(146., 200., 18.),
                          np.arange(200., 800., 24.)))
alt_bins = [[0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18],
            [19, 20, 22, 23, 24],
            [25, 26, 27, 28, 29]]

# ----------------------------------------------------------------------------
# Lidar BV Constants
# ----------------------------------------------------------------------------
lidar_bv_names = ['Tn_bac', 'dTn_bac', 'Nn_bac', 'dNn_bac']
n_lidar_bv = len(lidar_bv_names)
max_alt_lidar = 20

lidar_bv_mu = np.array([5.4422550, 1.10414981,
                        0.14230881, 0.000443083])
lidar_bv_sigma = np.array([0.0819466, 0.6514002,
                           0.1760367, 0.0003038])
lidar_thresholds = np.array(
    [[100, 324.35693359375],
     [-1, 50.0],
     [-1, 0.9010207056999207],
     [-1, 0.0025]])

alt_bins_lidar = [[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8, 9],
                  [10, 11, 12],
                  [14, 15, 16],
                  [17, 18, 19]]

# ----------------------------------------------------------------------------
# HFP Constants
# ----------------------------------------------------------------------------
hfp_names = ['GW_tau', 'GW_lamh', 'GW_lamv', 'GW_ch', 'GW_cv', 'GW_phi', 'GW_hl', 'GW_hu']
hfp_full_names = ['Period',
                  r'$\lambda_\mathrm{horizontal}$', r'$\lambda_\mathrm{vertical}$',
                  r'$\mathrm{Horizontal\/}v_p$', r'$\mathrm{Vertical\/}v_p$',
                  'Horizontal Dir', r'$Alt_\mathrm{lower}$', r'$Alt_\mathrm{upper}$', ]
hfp_units = ['$min$', '$km$', '$km$', '$m/s$', '$m/s$', '$degree$', '$km$', '$k$m']
n_hfp = len(hfp_names)

n_waves = 1
log_hfp = [0, 1, 2, 3, 4]
invert_hfp = [2, 4]
hfp_z_ranges = [(-2.4, 4), (-4, 4), (-4, 4), (-4, 4), (-4, 4), (-1.5, 1.8), (-0.7, 4), (-4, 1.5)]
hfp_meas_ranges = [(0, 125), (0, 4000), (-4000, 0), (0, 2000),
                   (-2000, 0), (-180, 180), (175, 210), (220, 325)]

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

# ----------------------------------------------------------------------------
# Driver Constants
# ----------------------------------------------------------------------------
driver_names = ['Ap', 'F10.7', 'F10.7avg',
                'MEI', 'MLT', 'RMM1', 'RMM2',
                'SLT', 'SZA', 'ShadHeight',
                'T10', 'T100', 'T2', 'T30', 'T5', 'T70',
                'TCI', 'U10', 'U100', 'U2', 'U30', 'U5', 'U70',
                'ap', 'dst', 'moon_phase', 'moon_x', 'moon_y', 'moon_z']
driver_names_full = [r'Daily Ap Index', r'F10.7 Solar Radio Flux', r'Avg F10.7 Solar Radio Flux',
                     f'Multivariate ESNO Index', r'Magnetic Local Time', r'First MJO Index', r'Second MJO Index',
                     r'Solar Local Time', r'Solar Zenith Angle', r'ShadHeight',
                     r'Stratospheric Temp 10hPa', r'Stratospheric Temp 100hPa', r'Stratospheric Temp 2hPa',
                     r'Stratospheric Temp 30hPa', r'Stratospheric Temp 5hPa', r'Stratospheric Temp 70hPa',
                     r'Thermosphere Climate Index', r'Zonal Winds 10hPa', r'Zonal Winds 100hPa', r'Zonal Winds 2hPa',
                     r'Zonal Winds 30hPa', r'Zonal Winds 5hPa', r'Zonal Winds 70hPa',
                     r'ap Geomagnetic Index', r'Disturbance Storm Time', r'Moon Phase', f'Lunar x Position',
                     f'Lunar y Position', f'Lunar z Position']
driver_units = [r'Index Value', r'Flux Value', r'Flux Value',
                r'Index Value', r'Hour', r'Index Value', r'Index Value',
                r'Hour', r'Degree', 'Meters',
                r'Temp', r'Temp', r'Temp', r'Temp', r'Temp', r'Temp',
                r'Index Value', r'Velocity', r'Velocity', r'Velocity', r'Velocity', r'Velocity', r'Velocity',
                r'Index Value', r'Index Value', r'Degree', f'ECEF Position', f'ECEF Position', f'ECEF Position']
dr_cols = {}
for i in range(len(driver_names)):
    dr_cols[driver_names[i]] = i

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

log_driver_feats = ['Ap', 'F10.7', 'F10.7avg', 'SZA', 'ap', 'TCI']
driver_deltas = [0.05 for i in range(n_driver_feat)]

driver_mu = np.array([2.07743465e+00, 4.63693947e+00, 4.64421382e+00, -2.46207644e-01,
                      3.95687475e-05, -6.25173732e-04, 6.43372329e-03,
                      -8.12593367e-03, 6.05776452e-21, 2.20282346e-21, 4.47069527e+00,
                      3.11947567e+02, 2.23293515e+02, 2.19509480e+02, 2.48115106e+02,
                      2.17163253e+02, 2.32683039e+02, 2.18359199e+02, 2.52627520e+01,
                      7.50327066e+00, 5.37702514e+00, 9.95164914e+00, 5.97748061e+00,
                      8.52696810e+00, 5.41012999e+00, 1.94736060e+00, -1.21362302e+01,
                      -1.14102569e-03, 9.08751920e-04, -5.84762886e+02, 3.44262876e+02,
                      -8.97374765e+00])
driver_sigma = np.array([7.58401494e-01, 3.46959096e-01, 3.30082911e-01, 8.90664279e-01,
                         7.10046100e-01, 7.04154914e-01, 1.02639672e+00,
                         1.01361708e+00, 7.07106781e-01, 7.07106781e-01, 2.76368897e-01,
                         6.42401709e+02, 1.36209122e+01, 7.08975914e+00, 1.87231700e+01,
                         1.04838166e+01, 1.64043198e+01, 8.28530392e+00, 7.15184997e-01,
                         1.39146472e+01, 4.31546545e+00, 1.84914106e+01, 9.62279004e+00,
                         1.60456992e+01, 5.70533927e+00, 9.15405618e-01, 2.00200575e+01,
                         7.03866365e-01, 7.10330918e-01, 2.69984897e+05, 2.73810015e+05,
                         2.44661624e+04])

# ----------------------------------------------------------------------------
# TEC Constants
# ----------------------------------------------------------------------------
wtec_vars = ['Amp', 'Tau', 'Lambda', 'C1', 'C2', 'Phi']
wtec_names = [v + '_' + s for v in wtec_vars for s in ['min', 'avg', 'max']]
wtec_units = [v for v in [r"$utec$", r"$min$", r"$m$", r"$m/s$", r"$m/s$", r"$degree$"] for s in range(3)]
n_wtec_feat = len(wtec_names)
n_wtec = len(wtec_names)
log_wtec = []

wtec_dr_names = driver_names.copy()
wtec_dr_feat_names = driver_feat_names.copy()
n_wtec_dr_feat = len(wtec_dr_feat_names)
n_wtec_dr = len(wtec_dr_names)

wtec_sites = ['Poker', 'MtMoses']
wtec_datasets_names = ['SSTIDs_Poker',
                       'MSTIDs_Poker',
                       'LSTIDs_Poker',
                       'MSTIDs_MtMoses',
                       'LSTIDs_MtMoses']
wtec_datasets_names_full = ['SSTIDs_300-900_Poker',
                            'MSTIDs_600-3600_Poker',
                            'LSTIDs_3600-7200_Poker'
                            'MSTIDs_600-3600_MtMoses',
                            'LSTIDs_3600-7200_MtMoses']
wtec_default_dataset = 'LSTIDs_Poker'
wtec_default_model = f'wtec_gan_{wtec_default_dataset}'

wtec_zscale_dict = {
    'SSTIDs_Poker':
        {
            'mu': np.array([1.49493658e-01, 1.49493658e-01, 1.49493658e-01, 1.11572285e+01,
                            1.23790008e+01, 1.40101894e+01, 2.24706428e+01, 2.80838807e+01,
                            3.54034051e+01, 1.45015018e+02, 2.85128797e+02, 6.44353012e+02,
                            1.58412218e+02, 1.58412218e+02, 1.58412218e+02, 1.25653259e+02,
                            1.25653259e+02, 1.25653259e+02]),
            'sigma': np.array([1.60593979e-01, 1.60593979e-01, 1.60593979e-01, 1.30968654e+00,
                               1.29393073e+00, 2.44538212e+00, 6.65094045e+00, 6.90048902e+00,
                               1.08743472e+01, 1.63567147e+02, 2.09199064e+02, 4.59782809e+02,
                               5.09746017e+01, 5.09746017e+01, 5.09746017e+01, 4.36278531e+01,
                               4.36278531e+01, 4.36278531e+01]),
            'meas_ranges': np.array([[0, 0.75], [0, 0.75], [0, 0.75],
                                     [7, 18], [8, 18], [8, 25],
                                     [8, 50], [10, 60], [10, 80],
                                     [0, 600], [0, 1100], [0, 2000],
                                     [15, 370], [15, 370], [15, 370],
                                     [12, 225], [12, 225], [12, 225]]),
        },
    'MSTIDs_Poker':
    {
        'mu': np.array([2.42431142e-01, 2.42484003e-01, 2.42425000e-01, 3.92242794e+01,
                        3.61509973e+01, 4.82875812e+01, 1.47413905e+02, 1.08307001e+02,
                        1.90002193e+02, 6.85971064e+02, 3.97044451e+02, 1.79607795e+03,
                        4.94861521e+01, 4.94826155e+01, 4.94722366e+01, 1.15135035e+02,
                        1.15134760e+02, 1.15116366e+02]),
        'sigma': np.array([2.59663941e-01, 2.59775508e-01, 2.59857762e-01, 6.46888822e+00,
                           4.46572969e+00, 7.00019071e+00, 5.18798655e+01, 2.17202072e+01,
                           5.46769045e+01, 6.27161838e+02, 1.64774301e+02, 7.81890685e+02,
                           1.00175222e+01, 1.00351606e+01, 1.00217572e+01, 3.26329293e+01,
                           3.26985695e+01, 3.26763362e+01]),
        'meas_ranges': np.array([[0, 1], [0, 1], [0, 1],
                                 [20, 60], [20, 55], [30, 70],
                                 [25, 325], [25, 200], [25, 375],
                                 [0, 3250], [0, 1500], [0, 5000],
                                 [20, 80], [20, 80], [20, 80],
                                 [22, 210], [22, 210], [22, 210]]),
    },
    'LSTIDs_Poker':
        {
            'mu': np.array([4.23489436e-01, 4.23466047e-01, 4.23461976e-01, 7.41123211e+01,
                            7.17588309e+01, 8.25004200e+01, 2.37354213e+02, 2.24137804e+02,
                            2.77067807e+02, 1.09877892e+03, 9.82959971e+02, 2.45849082e+03,
                            5.10621446e+01, 5.10610150e+01, 5.10643138e+01, 1.40271510e+02,
                            1.40284328e+02, 1.40289663e+02]),
            'sigma': np.array([3.46243800e-01, 3.46243892e-01, 3.46227740e-01, 9.14305952e+00,
                               6.02390689e+00, 1.09599962e+01, 5.05155109e+01, 3.88970463e+01,
                               6.00695246e+01, 8.92114810e+02, 5.05397785e+02, 1.03002496e+03,
                               7.22393651e+00, 7.22592818e+00, 7.22106388e+00, 3.55417336e+01,
                               3.55486743e+01, 3.55392444e+01]),
            'meas_ranges': np.array([[0, 2], [0, 2], [0, 2],
                                     [60, 100], [60, 95], [60, 105],
                                     [100, 400], [100, 350], [100, 450],
                                     [0, 4000], [0, 3000], [0, 5500],
                                     [25, 80], [25, 80], [25, 80],
                                     [25, 210], [25, 210], [25, 210]]),
        },
    'MSTIDs_MtMoses':
        {
            'mu': np.array([1.13727642e-01, 1.13727642e-01, 1.13729278e-01, 3.69019754e+01,
                            3.69552303e+01, 4.50127516e+01, 6.10248917e+01, 6.01530972e+01,
                            8.15851855e+01, 2.44924900e+02, 2.52104244e+02, 7.67086514e+02,
                            2.71683511e+01, 2.71683511e+01, 2.71681545e+01, 8.18615572e+01,
                            8.18615572e+01, 8.18612482e+01]),
            'sigma': np.array([6.83412430e-02, 6.83412430e-02, 6.83446761e-02, 5.81368005e+00,
                               4.73249178e+00, 6.01115417e+00, 1.51178371e+01, 1.18790041e+01,
                               1.52702505e+01, 2.01954718e+02, 1.23997927e+02, 3.17346901e+02,
                               4.99354555e+00, 4.99354555e+00, 4.99351532e+00, 4.20021174e+01,
                               4.20021174e+01, 4.20023988e+01]),
            'meas_ranges': np.array([[0, 0.5], [0, 0.5], [0, 0.5],
                                     [22, 58], [22, 53], [30, 65],
                                     [20, 130], [25, 115], [25, 150],
                                     [0, 1200], [10, 800], [10, 2000],
                                     [12, 45], [12, 45], [12, 45],
                                     [0, 200], [0, 200], [0, 200]]),
        },
    'LSTIDs_MtMoses':
        {
            'mu': np.array([2.48867082e-01, 2.48867082e-01, 2.48867082e-01, 6.44417946e+01,
                            6.51024296e+01, 6.63471480e+01, 1.04470923e+02, 1.07142481e+02,
                            1.11860912e+02, 5.18252583e+02, 6.20468237e+02, 8.52300798e+02,
                            2.76722976e+01, 2.76722976e+01, 2.76722976e+01, 8.07346327e+01,
                            8.07346327e+01, 8.07346327e+01]),
            'sigma': np.array([1.91078087e-01, 1.91078087e-01, 1.91078087e-01, 4.30263011e+00,
                               4.06091637e+00, 5.09518041e+00, 1.59912301e+01, 1.44015237e+01,
                               1.47095048e+01, 4.01924535e+02, 3.75192274e+02, 4.63746254e+02,
                               4.48856592e+00, 4.48856592e+00, 4.48856592e+00, 2.35016815e+01,
                               2.35016815e+01, 2.35016815e+01]),
            'meas_ranges': np.array([[0, 1], [0, 1], [0, 1],
                                     [60, 73], [60, 73], [60, 75],
                                     [58, 148], [63, 148], [75, 148],
                                     [0, 1600], [0, 1700], [0, 2100],
                                     [15, 40], [15, 40], [15, 40],
                                     [25, 130], [25, 130], [25, 130]]),
        },
}
for n in wtec_zscale_dict.keys():
    wtec_zscale_dict[n]['z_ranges'] = np.zeros((n_wtec, 2))
    for i in range(n_wtec):
        wtec_zscale_dict[n]['z_ranges'][i] = (wtec_zscale_dict[n]['meas_ranges'][i]
                                              - wtec_zscale_dict[n]['mu'][i]) / wtec_zscale_dict[n]['sigma'][i]

wtec_driver_mu = driver_mu.copy()
wtec_driver_sigma = driver_sigma.copy()

wtec_avg_coefficients = [0.05, 0.05, 0.05,
                         0.1, 0.1, 0.1,
                         0.05, 0.05, 0.05,
                         0.05, 0.05, 0.05,
                         0.05, 0.05, 0.05,
                         0.05, 0.05, 0.05]
wtec_thresholds = np.ones((n_wtec, 2))
wtec_thresholds[:, 0], wtec_thresholds[:, 1] = -np.inf, np.inf

# ----------------------------------------------------------------------------
# Driver Estimation constants
# ----------------------------------------------------------------------------
dr_delay = 2 * 3600  # 2 hours

# ----------------------------------------------------------------------------
# Hellinger Distance Constants
# ----------------------------------------------------------------------------
bin_exp = 55 / 100
filter_exp = 1 / 3

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

# assert ((n_wtec,) == wtec_mu.shape)
# assert ((n_wtec,) == wtec_sigma.shape)
assert ((n_wtec_dr_feat,) == wtec_driver_mu.shape)
assert ((n_wtec_dr_feat,) == wtec_driver_sigma.shape)
assert ((n_wtec, 2) == wtec_thresholds.shape)
