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
for name in driver_names + ["MLAT", "MLON"]:
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

wtec_sites = {'PFRR': dict(lat=65.124313, lon=-147.487866),
              'MtMoses': dict(lat=40.145285, lon=-117.415938),
              'NMexicoSPA': dict(lat=32.990278, lon=-106.969722),
              'ColoradoKCFO': dict(lat=39.785278, lon=-104.543056),
              'OklahomaCSM': dict(lat=35.3530, lon=-99.1952),
              'HoustonTxSS': dict(lat=39.71006393, lon=-95.158888),
              'MidlandTxSS': dict(lat=40.90293121337890, lon=-102.201944)}
wtec_regions = {"SouthWest": ["NMexicoSPA", "ColoradoKCFO", "OklahomaCSM",
                              "HoustonTxSS", "MidlandTxSS"]}

wtec_tid_types = ['TSTIDs', 'SSTIDs', 'MSTIDs', 'LSTIDs']
wtec_dataset_names = ['TSTIDs_PFRR', 'SSTIDs_PFRR', 'MSTIDs_PFRR', 'LSTIDs_PFRR',
                      'MSTIDs_MtMoses', 'LSTIDs_MtMoses',
                      'TSTIDs_NMexicoSPA', 'SSTIDs_NMexicoSPA',
                      'TSTIDs_ColoradoKCFO', 'SSTIDs_ColoradoKCFO',
                      'TSTIDs_OklahomaCSM', 'SSTIDs_OklahomaCSM',
                      'TSTIDs_HoustonTxSS', 'SSTIDs_HoustonTxSS',
                      'TSTIDs_MidlandTxSS', 'SSTIDs_MidlandTxSS',
                      'TSTIDs_SouthWest', 'SSTIDs_SouthWest'
                      ]

wtec_default_location = 'PFRR'
wtec_default_tid_type = 'SSTIDs'
wtec_default_dataset = f'{wtec_default_tid_type}_{wtec_default_tid_type}'
wtec_default_model = f'wtec_gan_{wtec_default_dataset}'
wtec_current_version = 'v12.00'
wtec_vars = ['uAmp', 'Amp', 'Tau', 'Lambda', 'C1', 'Phi']
wtec_names = [v + '_' + s for v in wtec_vars for s in ['min', 'avg', 'max']]
wtec_units = [v for v in [r"$utec$", r"$utec$", r"$min$", r"$m$", r"$m/s$", r"$degree$"] for s in range(3)]
n_wtec = len(wtec_names)

wtec_dr_names = driver_names + ["MLAT", "MLON"]
wtec_dr_feat_names = driver_feat_names + ["MLAT", "MLON"]
wtec_driver_mu = np.hstack((driver_mu,
                            [np.mean([wtec_sites[loc]['lat'] for loc in wtec_sites.keys()])],
                            [np.mean([wtec_sites[loc]['lon'] for loc in wtec_sites.keys()])]))
wtec_driver_sigma = np.hstack((driver_sigma,
                               [np.std([wtec_sites[loc]['lat'] for loc in wtec_sites.keys()])],
                               [np.std([wtec_sites[loc]['lon'] for loc in wtec_sites.keys()])]))
n_wtec_dr_feat = len(wtec_dr_feat_names)
n_wtec_dr = len(wtec_dr_names)

wtec_dict = {
    'TSTIDs':
        {
            'mu': np.array([9.38636408e-02, 2.05607027e-01, 3.85641354e-01, 1.45080889e+01,
                            1.97442527e+01, 2.48496753e+01, 3.71287834e+00, 5.17672456e+00,
                            6.52194381e+00, 8.46739282e+00, 1.48660460e+01, 2.70651936e+01,
                            1.39284072e+02, 2.84379932e+02, 7.12111779e+02, -1.18905920e+02,
                            -5.27365806e+01, 5.91004622e+01]),
            'sigma': np.array([6.22644776e-02, 8.14484467e-02, 1.71388731e-01, 1.05141952e+01,
                               1.31642593e+01, 1.60055103e+01, 6.64501815e-01, 4.00782214e-01,
                               5.34528379e-01, 3.09150037e+00, 4.20969972e+00, 1.12629783e+01,
                               5.51504920e+01, 1.24296504e+02, 4.59571438e+02, 8.93409712e+01,
                               9.94693317e+01, 9.98704445e+01]),
            'meas_ranges': np.array([[3.00000452e-02, 4.39926696e-01],
                                     [3.00443565e-02, 5.54520451e-01],
                                     [3.00443565e-02, 1.17944523e+00],
                                     [5.28477252e-01, 6.12088336e+01],
                                     [1.30425131e+00, 7.69977681e+01],
                                     [1.30425131e+00, 9.25292827e+01],
                                     [2.00021985e+00, 5.92390623e+00],
                                     [3.79173456e+00, 6.54762096e+00],
                                     [4.48596638e+00, 7.86505941e+00],
                                     [3.16568756e-01, 2.34387021e+01],
                                     [4.84191506e+00, 3.87642457e+01],
                                     [3.78581203e+00, 7.00730790e+01],
                                     [5.81393719e+00, 4.74665438e+02],
                                     [3.75185434e+01, 1.08349261e+03],
                                     [2.74125137e+01, 2.57406772e+03],
                                     [-1.79519531e+02, 1.78894150e+02],
                                     [-1.76307481e+02, 1.78894150e+02],
                                     [-1.73444414e+02, 1.79731003e+02]]),
            'avg_coefficients': [0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05],
            'period': "100-500sec",
        },
    'SSTIDs':
        {
            'mu': np.array([8.36454119e-02, 1.78325263e-01, 3.56570573e-01, 1.41787906e+01,
                            1.96426335e+01, 2.50497913e+01, 7.34631206e+00, 1.00020157e+01,
                            1.23136911e+01, 1.50462709e+01, 2.71894172e+01, 4.91214666e+01,
                            1.82833288e+02, 3.52773464e+02, 8.14990216e+02, -1.22101732e+02,
                            -5.44723696e+01, 7.00522547e+01]),
            'sigma': np.array([5.13713718e-02, 7.60288728e-02, 1.63301730e-01, 1.02268022e+01,
                               1.28818279e+01, 1.58768816e+01, 1.08272865e+00, 6.60386280e-01,
                               8.08633632e-01, 5.46429125e+00, 6.11921871e+00, 1.57192014e+01,
                               6.79737007e+01, 1.05929378e+02, 4.24344284e+02, 8.73666699e+01,
                               9.95375063e+01, 9.60240869e+01]),
            'meas_ranges': np.array([[3.00002066e-02, 4.02784961e-01],
                                     [3.01355209e-02, 6.24363727e-01],
                                     [3.01355209e-02, 1.23549753e+00],
                                     [5.25698841e-01, 5.88713665e+01],
                                     [1.26294112e+00, 7.55965338e+01],
                                     [1.26294112e+00, 9.25146589e+01],
                                     [5.00027262e+00, 1.09884306e+01],
                                     [7.62428917e+00, 1.20689910e+01],
                                     [9.05398952e+00, 1.41572347e+01],
                                     [4.75101685e-01, 3.83416357e+01],
                                     [1.18357832e+01, 5.86009976e+01],
                                     [1.04458843e+01, 1.05347048e+02],
                                     [8.86924553e+00, 5.40623177e+02],
                                     [1.05783891e+02, 1.05715974e+03],
                                     [4.92665138e+01, 2.39552470e+03],
                                     [-1.79519531e+02, 1.78844971e+02],
                                     [-1.75594520e+02, 1.78844971e+02],
                                     [-1.52718806e+02, 1.79625824e+02]]),
            'avg_coefficients': [0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05],
            'period': "300-900sec",
        },
    'MSTIDs':
        {
            'mu': np.array([1.40149274e-01, 2.58495638e-01, 4.77684164e-01, 1.05840930e+01,
                            1.30218616e+01, 1.59608845e+01, 2.48962676e+01, 3.62100279e+01,
                            4.77392048e+01, 5.10643164e+01, 8.38334650e+01, 1.20878816e+02,
                            1.90948441e+02, 3.19877951e+02, 5.15433438e+02, -5.52900145e+01,
                            1.89239784e+01, 1.00177410e+02]),
            'sigma': np.array([1.06592396e-01, 1.69129990e-01, 2.76081979e-01, 6.64096706e+00,
                               7.40413516e+00, 8.65392530e+00, 5.67354991e+00, 5.06483403e+00,
                               5.71753332e+00, 1.90208814e+01, 2.41296610e+01, 3.78049352e+01,
                               6.41000644e+01, 8.78925597e+01, 1.70410665e+02, 1.22793573e+02,
                               1.32505200e+02, 1.05006383e+02]),
            'meas_ranges': np.array([[3.00004921e-02, 6.68886168e-01],
                                     [3.11150294e-02, 1.00305401e+00],
                                     [3.12933589e-02, 1.53878331e+00],
                                     [5.28477252e-01, 3.54318237e+01],
                                     [1.01215839e+00, 3.91044399e+01],
                                     [1.01215839e+00, 4.64536691e+01],
                                     [1.50023234e+01, 4.81585579e+01],
                                     [2.17330526e+01, 4.96088000e+01],
                                     [3.04311342e+01, 6.43355298e+01],
                                     [9.35620663e+00, 1.14764822e+02],
                                     [3.48046818e+01, 1.60370936e+02],
                                     [4.76754432e+01, 2.26812164e+02],
                                     [4.60377769e+01, 4.78309139e+02],
                                     [1.01839938e+02, 6.50641224e+02],
                                     [1.35538838e+02, 1.11995543e+03],
                                     [-1.78864426e+02, 1.78359894e+02],
                                     [-1.77118668e+02, 1.78359894e+02],
                                     [-1.56463391e+02, 1.79006790e+02]]),
            'avg_coefficients': [0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05],
            'period': "900-3600sec",
        },
    'LSTIDs':
        {
            'mu': np.array([3.20131120e-01, 3.68809542e-01, 4.03491001e-01, 1.20627597e+01,
                            1.23216911e+01, 1.25821854e+01, 6.36437841e+01, 6.41107991e+01,
                            6.46687106e+01, 1.51315189e+02, 1.58050305e+02, 1.63622049e+02,
                            4.82493066e+02, 5.03812327e+02, 5.25263617e+02, 1.09447533e+02,
                            1.16878483e+02, 1.22930420e+02]),
            'sigma': np.array([0.2221592, 0.24864885, 0.25716501, 6.87636982,
                               6.90175715, 6.97461903, 5.1917861, 5.30732068,
                               5.81947416, 36.12762479, 36.75896196, 40.23401408,
                               121.21987889, 123.71468743, 132.81188331, 114.16182978,
                               105.32673265, 102.46354306]),
            'meas_ranges': np.array([[3.00596356e-02, 1.27714083e+00],
                                     [3.06534561e-02, 1.62278449e+00],
                                     [3.06534561e-02, 1.44047037e+00],
                                     [2.07009864e+00, 4.07136223e+01],
                                     [2.10402775e+00, 4.07280761e+01],
                                     [2.10402775e+00, 4.07280761e+01],
                                     [6.00092936e+01, 1.02628431e+02],
                                     [6.00092936e+01, 1.02628431e+02],
                                     [6.00092936e+01, 1.03118969e+02],
                                     [3.68539895e+01, 2.35869083e+02],
                                     [5.39586113e+01, 2.38734131e+02],
                                     [4.41925595e+01, 2.43909194e+02],
                                     [1.20904726e+02, 7.63121042e+02],
                                     [1.37441959e+02, 7.87221027e+02],
                                     [1.52583135e+02, 8.13955315e+02],
                                     [-1.73254303e+02, 1.74781311e+02],
                                     [-1.72396744e+02, 1.74938049e+02],
                                     [-1.72362396e+02, 1.75161804e+02]]),
            'avg_coefficients': [0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05,
                                 0.05, 0.05, 0.05],
            'period': "3600-7200sec",
        }
}
# Derive z-scale ranges and threshold filtering for each dataset
for n in wtec_dict.keys():
    wtec_dict[n]['z_ranges'] = np.zeros((len(wtec_names), 2))
    for i in range(len(wtec_names)):
        wtec_dict[n]['z_ranges'][i] = (wtec_dict[n]['meas_ranges'][i]
                                       - wtec_dict[n]['mu'][i]) / wtec_dict[n]['sigma'][i]
    wtec_dict[n]['thresholds'] = np.stack((np.ones(len(wtec_names)) * -np.inf,
                                           np.ones(len(wtec_names)) * np.inf), axis=-1)
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

assert ((n_wtec_dr_feat,) == wtec_driver_mu.shape)
assert ((n_wtec_dr_feat,) == wtec_driver_sigma.shape)
