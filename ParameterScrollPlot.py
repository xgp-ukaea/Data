import numpy as np
from dms import general_tools as gt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from PIL import Image

def unc_plot(x, data, color='r', apply_smooth=True, label=None, legend=True):
    if apply_smooth:
        datan = gt.inpaint_nans(data)
        y = gt.smooth(np.ravel(datan[:,1]))
        yL = gt.smooth(np.ravel(datan[:,0]))
        yU = gt.smooth(np.ravel(datan[:,2]))
    else:
        y = np.ravel(data[:,1])
        yL = np.ravel(data[:,0])
        yU = np.ravel(data[:,2])
    plt.plot(np.ravel(x), y, color=color, linewidth=2, label=label)
    plt.fill_between(np.ravel(x), yL, yU, alpha=0.2, antialiased=True, color=color)
    if legend:
        plt.legend()

file = 'outputScroll'
output = np.load(file + '.npy', allow_pickle=True)
output = output[()]

L_c = np.arange(0, np.shape(output['ResultMC']['DenMC'])[0])
time = output['input']['Time']

density = output['ResultMC']['DenMC']
hot_temp = output['ResultMC']['TeEMC']
cold_temp = output['ResultMC']['TeRMC']
n_frac = output['ResultMC']['noneMC']
f_mol = output['ResultMC']['fmolMC']
path_length = output['ResultMC']['DLMC']

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 18), sharex='col')

indx_d = [0]  # Use a list to store indx_d as a mutable object


def update_LoS_plot(indx_d):
    # Subplot 1: Hot Temperature
    plt.sca(ax1)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(hot_temp[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='r', label='Hot Temp')
    ax1.set_ylabel('Temp / eV')
    ax1.set_ylim(0, 5)
    ax1.grid()

    # Subplot 3: Cold Temperature
    plt.sca(ax2)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(cold_temp[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='b', label='Cold Temp')
    ax2.set_ylabel('Temp / eV')
    ax2.set_ylim(0, 5)
    ax2.grid()

    # Subplot 3: Density
    plt.sca(ax3)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(density[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='g', label='Density')
    ax3.set_ylabel('Particles / m^2 / s')
    ax3.set_ylim(1e19, 2e20)
    ax3.grid()

    # Subplot 4: Neutral Particle Fraction
    plt.sca(ax4)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(n_frac[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='purple', label='Neutral Particle Fraction')
    ax4.set_ylabel('n_0 / n_e')
    ax4.set_xlabel('Line of Sight')
    ax4.set_ylim(0, 1)
    ax4.grid()

    # Subplot 5: Molecule Fraction
    plt.sca(ax5)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(f_mol[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='orange', label='f_mol')
    ax5.set_ylabel('f_mol')
    ax5.set_xlabel('Line of Sight')
    ax5.set_ylim(0, 1)
    ax5.grid()

    # Subplot 6: Path Length
    plt.sca(ax6)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(path_length[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='cyan', label='Path Length')
    ax6.set_ylabel('Path Length')
    ax6.set_xlabel('Line of Sight')
    ax6.set_ylim(0, 0.4)
    ax6.grid()

    # Set the title for the whole figure
    fig.suptitle(f'Time = {time[indx_d[0]]}')


# Initialize the plot
update_LoS_plot(indx_d)

# Create the scrollbar
axscroll = plt.axes([0.1, 0.03, 0.8, 0.02])
scroll = Slider(axscroll, 'Time', 0, len(output['input']['Time']) - 1, valinit=indx_d[0], valstep=1)

def update(val):
    indx_d[0] = int(scroll.val)
    update_LoS_plot(indx_d)

scroll.on_changed(update)

def SaveAnimation():
    # Create the FuncAnimation instance with your figure and update function
    animation = FuncAnimation(fig, update, frames=len(output['input']['Time']), interval=200)

    # Save the animation frames as images using Pillow
    frames = []
    for i in range(len(output['input']['Time'])):
        indx_d[0] = i
        update_LoS_plot(indx_d)
        fig.canvas.draw()  # Update the canvas
        frame = np.array(fig.canvas.renderer.buffer_rgba())  # Convert canvas to image
        frames.append(Image.fromarray(frame))

    # Save the frames as a GIF file
    frames[0].save('SimPostProcessParams.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)


# Display the plot (optional, you can comment this line if you only want to save the animation)
plt.show()
