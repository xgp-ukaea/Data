import numpy as np
from dms import general_tools as gt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from PIL import Image
import dms.analysis.emission.Balmer_analysis_bayes as BA

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


def ImportEmiss(output, band, indx):
    mol = output['EmissMC'][band + 'MolMC']
    exc = output['EmissMC'][band + 'ExcMC']
    rec = output['EmissMC'][band + 'RecMC']
    total = mol + exc + rec

    return mol, exc, rec, total


def PlotEmiss(ax, mol, exc, rec, total, measured, indx_d):
    plt.sca(ax)
    plt.cla()
    unc_plot(L_c, np.transpose(np.nanquantile(mol[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='r', label='Molecular')
    unc_plot(L_c, np.transpose(np.nanquantile(exc[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='g', label='Excitation')
    unc_plot(L_c, np.transpose(np.nanquantile(rec[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='b', label='Recombination')
    unc_plot(L_c, np.transpose(np.nanquantile(total[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='black', label='Total Emission')
    plt.plot(L_c, measured[:, indx_d], color='black', linestyle='dashed', label='Measured Emission Intensity')
    plt.legend()
    ax.set_xlabel('Distance wrt(m)')
    ax.set_ylabel('Emission Intensity')

    ax.grid()

    return


output = np.load('46860_BaySPMI_low_Te_150623_output_new_pec_post.npy', allow_pickle=True)
output = output[()]
#output = BA.emiss_extrap_structure(output)
#np.save('46860_BaySPMI_low_Te_150623_output_new_pec_post.npy', output)

L_c = np.arange(0, np.shape(output['ResultMC']['Ion'])[0])
L_c = np.cumsum(np.ones(np.shape(L_c))*0.03)
time = output['input']['Time']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 18), sharex='col')

indx_d = [0]  # Use a list to store indx_d as a mutable object
Da_measured = output['input']['DaMea']
n1_measured = output['input']['n1Int']
n2_measured = output['input']['n2Int']

def update_LoS_plot(indx_d):
    Da_mol, Da_exc, Da_rec, Da_total = ImportEmiss(output, 'Da', indx_d)
    PlotEmiss(ax1, Da_mol, Da_exc, Da_rec, Da_total, Da_measured, indx_d)
    ax1.set_ylim(0, 8e21)
    ax1.set_title(f'Balmer-Alpha Emission at Time = {time[indx_d[0]]}')

    n1_mol, n1_exc, n1_rec, n1_total = ImportEmiss(output, 'n1', indx_d)
    PlotEmiss(ax2, n1_mol, n1_exc, n1_rec, n1_total, n1_measured, indx_d)
    ax2.set_ylim(0, 1.5e20)

    n2_mol, n2_exc, n2_rec, n2_total = ImportEmiss(output, 'n2', indx_d)
    PlotEmiss(ax3, n2_mol, n2_exc, n2_rec, n2_total, n2_measured, indx_d)
    ax3.set_ylim(0, 6e19)


# Initialize the plot
update_LoS_plot(indx_d)

# Create the scrollbar
axscroll = plt.axes([0.1, 0.05, 0.8, 0.03])
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
    frames[0].save('SimPostProcessLoS.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)


# Display the plot
plt.show()
