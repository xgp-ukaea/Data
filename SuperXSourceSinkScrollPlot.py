import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

output = np.load('outputScroll.npy', allow_pickle=True)
output = output[()]

R = output['input']['R']
Z = output['input']['Z']
ioni = output['ResultMC']['Ion'] + output['ResultMC']['mai_h2p']# + output['ResultMC']['mai_h2']
mar = output['ResultMC']['mar_h2p']# + output['ResultMC']['mar_hm']
eir = output['ResultMC']['EIR']

indx_d = [0]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 18), sharex='col')

def MAST(ax):
    inner_x = (0.27, 0.34, 1.09, 1.35, 1.73)
    inner_y = (-1, -1.31, -2.06, -2.06, -1.68)
    ax.plot(inner_x, inner_y, color='black')

    outer_x = (1.2, 0.86, 0.91, 1.56)
    outer_y = (-1, -1.35, -1.59, -1.57)
    ax.plot(outer_x, outer_y, color='black')


# Ionisation plot
ax1.set_position([0.05, 0.4, 0.25, 0.5])  # Set the position of the main plot
ionisation = np.transpose(np.nanquantile(ioni[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
ion_plot = ionisation[:, 1]
scatter_plot1 = ax1.scatter(R, Z, c=ion_plot, cmap='Reds', linewidth=5, s=400, marker='|')
ax1.grid()
ax1.set_xlabel('R')
ax1.set_ylabel('Z')

# Set the axis limits
ax1.set_ylim(-2.1, -1)
ax1.set_xlim(0.25, 1.75)

MAST(ax1)

# Create the colorbar
cax1 = fig.add_axes([0.05, 0.3, 0.25, 0.03])  # [left, bottom, width, height]
cbar1 = fig.colorbar(scatter_plot1, cax=cax1, orientation='horizontal')
cbar1.set_label('Mean Inferred Ionisation')

# MAR plot
ax2.set_position([0.375, 0.4, 0.25, 0.5])  # Set the position of the main plot
MAR = np.transpose(np.nanquantile(mar[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
mar_plot = MAR[:, 1]
scatter_plot2 = ax2.scatter(R, Z, c=mar_plot, cmap='Greens', linewidth=5, s=400, marker='|')
ax2.grid()
ax2.set_xlabel('R')
ax2.set_ylabel('Z')

# Set the axis limits
ax2.set_ylim(-2.1, -1)
ax2.set_xlim(0.25, 1.75)

MAST(ax2)

# Create the colorbar
cax2 = fig.add_axes([0.375, 0.3, 0.25, 0.03])  # [left, bottom, width, height]
cbar2 = fig.colorbar(scatter_plot2, cax=cax2, orientation='horizontal')
cbar2.set_label('Mean Inferred MAR')

# EIR plot
ax3.set_position([0.7, 0.4, 0.25, 0.5])  # Set the position of the main plot
EIR = np.transpose(np.nanquantile(eir[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
eir_plot = EIR[:, 1]
scatter_plot3 = ax3.scatter(R, Z, c=eir_plot, cmap='Blues', linewidth=5, s=400, marker='|')
ax3.grid()
ax3.set_xlabel('R')
ax3.set_ylabel('Z')

# Set the axis limits
ax3.set_ylim(-2.1, -1)
ax3.set_xlim(0.25, 1.75)

MAST(ax3)

# Create the colorbar
cax3 = fig.add_axes([0.7, 0.3, 0.25, 0.03])  # [left, bottom, width, height]
cbar3 = fig.colorbar(scatter_plot3, cax=cax3, orientation='horizontal')
cbar3.set_label('Mean Inferred EIR')


def update_SuperX_plot(indx_d):
    ionisation = np.transpose(np.nanquantile(ioni[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
    ion_plot = ionisation[:, 1]
    scatter_plot1.set_array(ion_plot)  # Update the color data in the scatter plot

    MAR = np.transpose(np.nanquantile(mar[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
    mar_plot = MAR[:, 1]
    scatter_plot2.set_array(mar_plot)

    EIR = np.transpose(np.nanquantile(eir[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1))
    eir_plot = EIR[:, 1]
    scatter_plot3.set_array(eir_plot)


# Create the scrollbar
axscroll = plt.axes([0.1, 0.05, 0.8, 0.03])
scroll = Slider(axscroll, 'Time', 0, len(output['input']['Time']) - 1, valinit=indx_d[0], valstep=1)


def update(val):
    indx_d[0] = int(scroll.val)
    update_SuperX_plot(indx_d)


scroll.on_changed(update)

# Show the plot
plt.show()
