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

# Apply post-processing:
#output = BA.postproc_structure(output)

L_c = np.arange(0, np.shape(output['ResultMC']['Ion'])[0])
time = output['input']['Time']
ioni = output['ResultMC']['Ion'] + output['ResultMC']['mai_h2p']
mar = output['ResultMC']['mar_h2p']

fig = plt.figure(figsize=(12, 8))
ax = plt.axes([0.1, 0.3, 0.8, 0.6])

indx_d = [0]


def update_LoS_plot(indx_d):
    plt.sca(ax)
    plt.cla()
    unc_plot(L_c, np.transpose(np.quantile(ioni[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='r', label='Ionisation')
    unc_plot(L_c, np.transpose(np.quantile(mar[:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='g', label='MAR')
    unc_plot(L_c, np.transpose(np.quantile(output['ResultMC']['EIR'][:, indx_d[0], :], [0.16, 0.5, 0.84], axis=1)), color='b', label='EIR')
    ax.set_xlabel('Line of Sight')
    ax.set_ylabel('Particles / m^2 / s')
    ax.set_title(f'Time = {time[indx_d[0]]}')
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
