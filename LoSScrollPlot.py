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

output = np.load('outputScroll.npy', allow_pickle=True)
output = output[()]
#output = BA.rates_extrap_adas_structure(output, low_te=False)
#output = BA.rates_extrap_mol_structure(output)
#np.save('SimulatedNewPECPost.npy', output)

L_c = np.arange(0, np.shape(output['ResultMC']['Ion'])[0])
L_c = np.cumsum(np.ones(np.shape(L_c))*0.03)
time = output['input']['Time']
ioni = output['ResultMC']['Ion'] + output['ResultMC']['mai_h2p']# + output['ResultMC']['mai_h2']
mar =  output['ResultMC']['mar_h2p']# + output['ResultMC']['mar_hm']

fig = plt.figure(figsize=(12, 8))

indx_d = [0]  # Use a list to store indx_d as a mutable object
ax = plt.axes([0.1, 0.3, 0.8, 0.6])

def get_SOLPS_output(output,indx):
    Ion_V1 = output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['Ion'] + output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['MAIH2']# + output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['MAIH2p']
    Ion_V2 = output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['Ion'] + output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['MAIH2']# + output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['MAIH2p']
    Ion_SOLPS = np.append(Ion_V1, Ion_V2)

    MAR_V1 = output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['MARH2p']# + output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['MARHm']
    MAR_V2 = output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['MARH2p']# + output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['MARHm']
    MAR_SOLPS = np.append(MAR_V1, MAR_V2)

    EIR_V1 = output['input']['miscel'][indx]['o_V1'][0]['DSS']['Model_Hydro']['IntegrParam']['Rec']
    EIR_V2 = output['input']['miscel'][indx]['o_V2'][0]['DSS']['Model_Hydro']['IntegrParam']['Rec']
    EIR_SOLPS = np.append(EIR_V1, EIR_V2)

    return Ion_SOLPS, MAR_SOLPS, EIR_SOLPS

def update_LoS_plot(indx_d):
    plt.sca(ax)
    plt.cla()
    unc_plot(L_c, np.transpose(np.nanquantile(ioni[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='r', label='Ionisation')
    unc_plot(L_c, np.transpose(np.nanquantile(mar[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='g', label='MAR')
    unc_plot(L_c, np.transpose(np.nanquantile(output['ResultMC']['EIR'][:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='b', label='EIR')
    # get solps
    ion_s, mar_s, eir_s = get_SOLPS_output(output,indx_d[0])
    plt.plot(L_c,ion_s,'r--',label='Ionisation SOLPS',linewidth=2)
    plt.plot(L_c, mar_s, 'g--', label='MAR SOLPS', linewidth=2)
    plt.plot(L_c, eir_s, 'b--', label='EIR SOLPS', linewidth=2)
    ax.set_xlabel('Distance wrt(m)')
    ax.set_ylabel('Particles / m^2 / s')
    ax.set_title(f'Time = {time[indx_d[0]]}')
    #ax.set_ylim(0, 6e22)
    ax.grid()

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