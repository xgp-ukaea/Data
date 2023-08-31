from agsi.models import BalmerTwoTemperature
import numpy as np
import matplotlib.pyplot as plt
from dms import general_tools as gt
from matplotlib.widgets import Slider

def unc_plot(x,data,color='r',apply_smooth=True, label=None, legend=True):
    import matplotlib.pyplot as plt
    if apply_smooth:
        datan = gt.inpaint_nans(data)
        y = gt.smooth(np.ravel(datan[:,1]))
        yL= gt.smooth(np.ravel(datan[:,0]))
        yU= gt.smooth(np.ravel(datan[:,2]))
    else:
        y = np.ravel(data[:,1])
        yL= np.ravel(data[:,0])
        yU= np.ravel(data[:,2])
    plt.plot(np.ravel(x),y,color=color,linewidth=2, label=label)
    plt.fill_between(np.ravel(x),yL,yU,alpha=0.2,antialiased=True,color=color)
    if legend:
        plt.legend()


output = np.load('46860_BaySPMI_low_Te_150623_output_new_pec_post.npy', allow_pickle=True)
output = output[()]

L_c = np.arange(0, np.shape(output['ResultMC']['Ion'])[0])
L_c = np.cumsum(np.ones(np.shape(L_c))*0.03)
time = output['input']['Time']

hot_temp = np.log(output['ResultMC']['TeEMC'])
cold_temp = np.log(output['ResultMC']['TeRMC'])
elec_dens = np.log(output['ResultMC']['DenMC'])
neut_frac = np.log(output['ResultMC']['noneMC'])
fmol = output['ResultMC']['fmolMC']
path_length = output['ResultMC']['DLMC']

# Create sample array with shape (30, 63, 500, 6)
results = np.array([hot_temp, cold_temp, elec_dens, neut_frac, fmol, path_length]).transpose(1, 2, 3, 0)

sample = results.reshape(-1, 6)

emission_model = BalmerTwoTemperature(lines=["D_alpha", "D_gamma", "D_delta"])

# Calculate sample_emission for all i and j combinations
sample_emission = emission_model.prediction(sample)

# Reshape sample_emission to the desired shape
sample_emission_reshaped = sample_emission.reshape(3, 30, 63, 500)

DaEmiss = sample_emission_reshaped[0, :, :, :]
n1Emiss = sample_emission_reshaped[1, :, :, :]
n2Emiss = sample_emission_reshaped[2, :, :, :]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 18), sharex='col')
indx_d = [0]
Da_measured = output['input']['DaMea']
n1_measured = output['input']['n1Int']
n2_measured = output['input']['n2Int']


def update_LoS_plot(indx_d):
    plt.sca(ax1)
    plt.cla()
    unc_plot(L_c, np.transpose(np.nanquantile(DaEmiss[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='black', label='Sample Emission Intensity')
    plt.plot(L_c, Da_measured[:, indx_d], color='black', linestyle='dashed', label='Measured Emission Intensity')
    plt.legend()
    ax1.set_xlabel('Distance wrt(m)')
    ax1.set_ylabel('Emission Intensity')
    ax1.set_title(f'Balmer-Alpha Emission at Time = {time[indx_d[0]]}')
    ax1.grid()

    plt.sca(ax2)
    plt.cla()
    unc_plot(L_c, np.transpose(np.nanquantile(n1Emiss[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='black', label='Sample Emission Intensity')
    plt.plot(L_c, n1_measured[:, indx_d], color='black', linestyle='dashed', label='Measured Emission Intensity')
    plt.legend()
    ax2.set_xlabel('Distance wrt(m)')
    ax2.set_ylabel('Emission Intensity')
    ax2.set_title(f'Balmer-Gamma Emission at Time = {time[indx_d[0]]}')
    ax2.grid()

    plt.sca(ax3)
    plt.cla()
    unc_plot(L_c, np.transpose(np.nanquantile(n2Emiss[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='black', label='Sample Emission Intensity')
    plt.plot(L_c, n2_measured[:, indx_d], color='black', linestyle='dashed', label='Measured Emission Intensity')
    plt.legend()
    ax3.set_xlabel('Distance wrt(m)')
    ax3.set_ylabel('Emission Intensity')
    ax3.set_title(f'Balmer-Delta Emission at Time = {time[indx_d[0]]}')
    ax3.grid()


# Initialize the plot
update_LoS_plot(indx_d)

# Create the scrollbar
axscroll = plt.axes([0.1, 0.05, 0.8, 0.03])
scroll = Slider(axscroll, 'Time', 0, len(output['input']['Time']) - 1, valinit=indx_d[0], valstep=1)

def update(val):
    indx_d[0] = int(scroll.val)
    update_LoS_plot(indx_d)

scroll.on_changed(update)


# Display the plot
plt.show()
