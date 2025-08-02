""" 
Use to generate a plot of fANOVA vs. LPI importances for experiments we executed on the cluster.
"""

from matplotlib import pyplot as plt
import numpy as np

from optimization_problem import configuration_space, METADATA_FILE

lpi_cluster_run = [('use_BN', 0.40326190166829634, 0.0), ('learning_rate_init', 0.24562271955262038, 0.04354445295565775), ('n_conv_layers', 0.18906472092620705, 0.0), ('n_channels_fc_0', 0.04891256150634212, 0.03178545453347112), ('batch_size', 0.032608353367073926, 0.03519646158503436), ('n_channels_conv_1', 0.029685198563205183, 0.024936686629045125), ('n_fc_layers', 0.026022872425812797, 0.005773126184169473), ('n_channels_fc_2', 0.017370975915194122, 0.005411401752320504), ('n_channels_conv_0', 0.00614031512323473, 0.0014110723002433712), ('n_channels_fc_1', 0.0013103809520133589, 0.005639122354516205), ('global_avg_pooling', 3.862026670512489e-18, 0.0), ('dropout_rate', 0.0, 0.0), ('kernel_size', 0.0, 0.0), ('n_channels_conv_2', 0.0, 0.0)]
fanova_cluster_run = [(('n_conv_layers',), 0.32746708438203476, 0.365421456420154), (('n_channels_conv_0',), 0.2096401192575314, 0.28711615629130016), (('use_BN',), 0.08136712705007793, 0.142195198974274), (('n_channels_fc_0',), 0.04523664397489868, 0.08145819528677238), (('batch_size',), 0.044180607092226876, 0.07289199920514199), (('n_channels_fc_0', 'use_BN'), 0.028969356892435707, 0.06299649592747919), (('n_conv_layers', 'use_BN'), 0.02356101655322666, 0.05154660660551614), (('n_channels_fc_1',), 0.021190871454860663, 0.03719975649289479), (('learning_rate_init',), 0.02102596638237734, 0.07495372082687636), (('n_channels_conv_0', 'n_conv_layers'), 0.019989213172428834, 0.03555904794576074), (('n_channels_conv_1',), 0.015045255423063765, 0.025646044435291417), (('batch_size', 'n_conv_layers'), 0.014246016651715624, 0.02458106925178868), (('batch_size', 'n_channels_conv_0'), 0.013427667434824894, 0.040526414758553404), (('n_channels_conv_0', 'use_BN'), 0.009483285736415343, 0.021487166724115558), (('n_conv_layers', 'n_channels_fc_1'), 0.009392131731278733, 0.02261950941611204), (('use_BN', 'n_channels_fc_1'), 0.008929313847526925, 0.024306225399390226), (('n_channels_conv_0', 'n_channels_conv_1'), 0.008698221970837179, 0.019474505532443703), (('use_BN', 'n_channels_conv_1'), 0.007633557642222034, 0.01674740634565575), (('learning_rate_init', 'n_fc_layers'), 0.007221547554519464, 0.02792035866921562), (('global_avg_pooling',), 0.007106616814392334, 0.023476704358392834), (('n_conv_layers', 'n_channels_fc_2'), 0.006781311907051899, 0.013811280782211745), (('n_channels_fc_0', 'n_conv_layers'), 0.005698268320599772, 0.010374330191400319), (('n_channels_conv_0', 'n_channels_fc_0'), 0.0050327162451227515, 0.008721193232546293), (('n_fc_layers',), 0.00407315576102449, 0.014870791475289002), (('n_channels_fc_2',), 0.003909796443660498, 0.007667376668687958), (('batch_size', 'n_channels_conv_1'), 0.002951271213852907, 0.010983278821140837), (('n_channels_conv_0', 'n_channels_fc_1'), 0.002823994557395186, 0.0074996686171582725), (('global_avg_pooling', 'n_channels_conv_0'), 0.0023203341915670524, 0.00699005164884675), (('n_channels_conv_0', 'n_fc_layers'), 0.0020034249436745658, 0.007498291990562409), (('batch_size', 'use_BN'), 0.001712443305210297, 0.004459626829870304), (('n_channels_fc_0', 'n_channels_conv_1'), 0.001571349801801237, 0.004498710491058607), (('global_avg_pooling', 'n_channels_fc_1'), 0.001455050832372694, 0.0038484058449451846), (('n_conv_layers', 'n_channels_conv_1'), 0.0014424970424228476, 0.004243196517509451), (('learning_rate_init', 'n_conv_layers'), 0.0012362996431998306, 0.00204963632039865), (('batch_size', 'n_fc_layers'), 0.001083361522801386, 0.0021515492999421276), (('batch_size', 'n_channels_fc_1'), 0.0010598454815122233, 0.0023653442420525353), (('batch_size', 'n_channels_fc_0'), 0.0008972700025552201, 0.0020302828920984906), (('use_BN', 'n_channels_fc_2'), 0.0007587116123014919, 0.0026557374731790533), (('learning_rate_init', 'use_BN'), 0.0006044506280201867, 0.0007563489545655874), (('n_channels_fc_0', 'n_channels_fc_1'), 0.00048280506935712204, 0.0013409207412790995), (('batch_size', 'global_avg_pooling'), 0.00032165705452593123, 0.0010243062386363197), (('global_avg_pooling', 'n_conv_layers'), 0.00032135146817555886, 0.0008620706974486348), (('learning_rate_init', 'n_channels_conv_0'), 0.00027079497831192397, 0.0005647597310769634), (('n_channels_fc_1', 'n_channels_fc_2'), 0.00022340690592290117, 0.0005840263359055349), (('batch_size', 'n_channels_fc_2'), 0.00019227544997537415, 0.00035313608649058617), (('n_channels_fc_0', 'n_channels_fc_2'), 0.0001777057027430358, 0.0006478747680343471), (('global_avg_pooling', 'use_BN'), 0.00015622607384870393, 0.00045384824883393086), (('batch_size', 'learning_rate_init'), 0.00012383519453829588, 0.0003894982671035003), (('global_avg_pooling', 'n_fc_layers'), 0.00010055776529240853, 0.0003894580601234123), (('n_channels_conv_1', 'n_channels_fc_1'), 8.791158029510498e-05, 0.00030793786442544983), (('global_avg_pooling', 'n_channels_fc_0'), 5.8561450593127654e-05, 0.000164684005207528), (('global_avg_pooling', 'n_channels_conv_1'), 4.764057606084607e-05, 0.00013604527652485282), (('n_fc_layers', 'use_BN'), 3.955253415632933e-05, 0.00014912908148078855), (('learning_rate_init', 'n_channels_fc_0'), 3.131453707002325e-05, 3.571879915361148e-05), (('learning_rate_init', 'n_channels_conv_1'), 2.9462380618096116e-05, 5.161949336386457e-05), (('n_conv_layers', 'n_fc_layers'), 2.8084332191360137e-05, 8.091484108588096e-05), (('learning_rate_init', 'n_channels_fc_2'), 2.6805949369007634e-05, 6.753767017053182e-05), (('n_fc_layers', 'n_channels_fc_1'), 1.503234403224576e-05, 4.670733377059392e-05), (('learning_rate_init', 'n_channels_fc_1'), 1.4834750298198876e-05, 1.9458524267930032e-05), (('n_channels_fc_0', 'n_fc_layers'), 1.3445700628640414e-05, 3.209302767925609e-05), (('n_fc_layers', 'n_channels_conv_1'), 5.700532686718454e-06, 1.5310433219838587e-05), (('global_avg_pooling', 'learning_rate_init'), 4.556499109307471e-06, 8.575700801694543e-06), (('n_channels_conv_0', 'n_channels_fc_2'), 1.28308214812147e-06, 4.613957047928041e-06), (('n_fc_layers', 'n_channels_fc_2'), 6.49165041836707e-07, 2.5135120115266845e-06), (('learning_rate_init', 'n_channels_conv_2'), 6.476986277912216e-07, 1.4853702426207782e-06), (('global_avg_pooling', 'n_channels_fc_2'), 8.393347143300035e-08, 2.888792055875944e-07), (('n_channels_conv_2', 'n_channels_fc_1'), 6.5295580480960454e-09, 2.3750623832118197e-08), (('n_channels_fc_0', 'n_channels_conv_2'), 2.7277865639654635e-09, 9.953056655857486e-09), (('n_channels_conv_2',), 1.8516778094970675e-09, 4.912552217480026e-09), (('use_BN', 'n_channels_conv_2'), 1.8380749518801902e-09, 4.917357315638281e-09), (('n_channels_conv_0', 'n_channels_conv_2'), 7.375782382755944e-10, 2.8278920054878244e-09), (('n_channels_conv_1', 'n_channels_conv_2'), 4.523606956385115e-10, 1.7468834092563336e-09), (('n_channels_conv_1', 'n_channels_fc_2'), 4.1254486981869414e-10, 1.431936622411756e-09), (('n_conv_layers', 'n_channels_conv_2'), 3.203346668979115e-10, 1.085002546684591e-09), (('batch_size', 'n_channels_conv_2'), 2.2471345187591947e-11, 8.549161270366146e-11), (('n_fc_layers', 'n_channels_conv_2'), 2.0388481236950742e-11, 7.896242796532013e-11), (('global_avg_pooling', 'n_channels_conv_2'), 4.417728318062391e-12, 1.7081931348283826e-11), (('n_channels_conv_2', 'n_channels_fc_2'), 1.9705109139257817e-15, 6.201661715257429e-15)]

cs = configuration_space()

lpi_hps, lpi_mean, lpi_std = zip(*lpi_cluster_run)
fanova_hps, fanova_mean, fanova_std = zip(*fanova_cluster_run)

lpi_mean = np.asarray(lpi_mean)

lpi_hps_to_keep = []
lpi_mean_to_keep = []
# some cleaning of data
for i, name in enumerate(lpi_hps):
    if name in ['dropout_rate', 'kernel_size']:
        pass
    else:
        lpi_hps_to_keep += [name]
        lpi_mean_to_keep += [lpi_mean[i]]
lpi_hps = lpi_hps_to_keep
lpi_mean = np.asarray(lpi_mean_to_keep)

highlight_lr = 0
fanova_hps_to_keep = []
fanova_mean_to_keep = []
for i, name in enumerate(fanova_hps):
    if len(name) == 1:
        fanova_hps_to_keep += [name[0]]
        fanova_mean_to_keep += [fanova_mean[i]]
        if name[0] =='learning_rate_init':
            highlight_lr = len(fanova_hps_to_keep) - 1
fanova_hps = fanova_hps_to_keep
fanova_mean = np.asarray(fanova_mean_to_keep)
fanova_mean = fanova_mean / fanova_mean.sum()

# decide what to plot:
# hps = fanova_hps
# mean = fanova_mean
hps = fanova_hps
mean = fanova_mean


# Create positions for the bars
positions = np.arange(len(hps))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 4), sharey=True) #, gridspec_kw={'aspect': 'equal'})

fig.suptitle('1st order HP importances from warmstart data', fontweight='bold')
# Create the bar plot
# Customize plot
ax1.bar(positions, fanova_mean, align='center')
ax1.bar(highlight_lr, fanova_mean[highlight_lr], align='center', color='orange')
ax1.set_title(r'fANOVA$^\ast$')
ax1.set_xticks(positions, [str(hp) for hp in fanova_hps], rotation=90)
ax1.set_ylabel('HP importances')

# LPI
ax2.bar(positions, lpi_mean, align='center', zorder=1)
ax2.set_title('LPI')
ax2.set_xticks(positions, [str(hp) for hp in lpi_hps], rotation=90)

# Show the plot
plt.tight_layout()
plt.show()
