import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib.patches as mpatches


def generateVectorPatternWithText_0o2_nonNormalized(layer,pattern,title='TMP_TITLE',text=1):
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    grid =  np.zeros((pattern[0]*pattern[1],2))
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            grid[i+j*pattern[0],0] = 0.2*pattern[1] - 0.2*(j) - 0.2
            grid[i+j*pattern[0],1] = 0.2*(i)
    grid2 = np.zeros(grid.shape)
    grid2[:,0] = grid[:,1]
    grid2[:,1] = grid[:,0]
    grid=grid2
    patches = []
    colors = []
    float_formatter = lambda x: "%.4f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    for i in range(pattern[0]*pattern[1]):
        colors.append(layer.get_weights()[0][0,0,:,i]/np.max(layer.get_weights()[0][0,0,:,i]))
        rect = mpatches.Rectangle(grid[i], 0.2, 0.2, fill=True, facecolor=colors[i])
        ax.add_patch(rect)
        patches.append(rect)
        ax.annotate(str(float_formatter(colors[i][0])), xy=(grid[i]+(0.005,0.072)), xytext=((grid[i]+(0.0052,0.0722))),fontsize=(180/pattern[0]),fontweight='bold')
        ax.annotate(str(float_formatter(colors[i][1])), xy=(grid[i]+(0.005,0.042)), xytext=((grid[i]+(0.0052,0.0422))),fontsize=(180/pattern[0]),fontweight='bold')
        ax.annotate(str(float_formatter(colors[i][2])), xy=(grid[i]+(0.005,0.012)), xytext=((grid[i]+(0.0052,0.0122))),fontsize=(180/pattern[0]),fontweight='bold')
    #plt.subplots_adjust(left=0, right=0.2*pattern[0], bottom=0, top=0.2*pattern[1],wspace=0, hspace=0)
    #plt.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5,wspace=0, hspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    ax.set_ylim([0,0.2*pattern[0]])
    ax.set_xlim([0,0.2*pattern[1]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #plt.show()
    plt.savefig(title+'_'+str(pattern)+'_WithText_0o2.eps')
    plt.savefig(title+'_'+str(pattern)+'_WithText_0o2.svg')
    plt.savefig(title+'_'+str(pattern)+'_WithText_0o2.pdf')
    plt.gcf().clear()
    return

def generateVectorPatternWithText_0o2(layer,pattern,text=1):
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    grid =  np.zeros((pattern[0]*pattern[1],2))
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            grid[i+j*pattern[0],0] = 0.2*pattern[1] - 0.2*(j) - 0.2
            grid[i+j*pattern[0],1] = 0.2*(i)
    grid2 = np.zeros(grid.shape)
    grid2[:,0] = grid[:,1]
    grid2[:,1] = grid[:,0]
    grid=grid2
    patches = []
    colors = []
    float_formatter = lambda x: "%.4f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    for i in range(pattern[0]*pattern[1]):
        colors.append(layer.get_weights()[0][0,0,:,i])
        rect = mpatches.Rectangle(grid[i], 0.2, 0.2, fill=True, facecolor=colors[i])
        ax.add_patch(rect)
        patches.append(rect)
        ax.annotate(str(float_formatter(colors[i][0])), xy=(grid[i]+(0.005,0.072)), xytext=((grid[i]+(0.0052,0.0722))),fontsize=(180/pattern[0]),fontweight='bold')
        ax.annotate(str(float_formatter(colors[i][1])), xy=(grid[i]+(0.005,0.042)), xytext=((grid[i]+(0.0052,0.0422))),fontsize=(180/pattern[0]),fontweight='bold')
        ax.annotate(str(float_formatter(colors[i][2])), xy=(grid[i]+(0.005,0.012)), xytext=((grid[i]+(0.0052,0.0122))),fontsize=(180/pattern[0]),fontweight='bold')
    #plt.subplots_adjust(left=0, right=0.2*pattern[0], bottom=0, top=0.2*pattern[1],wspace=0, hspace=0)
    #plt.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5,wspace=0, hspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    ax.set_ylim([0,0.2*pattern[0]])
    ax.set_xlim([0,0.2*pattern[1]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #plt.show()
    plt.savefig(str(pattern)+'_WithText_0o2.eps')
    plt.savefig(str(pattern)+'_WithText_0o2.svg')
    plt.gcf().clear()
    return


def generateNDFAPattern(weights,pattern,text=1):
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    grid =  np.zeros((pattern[0]*pattern[1],2))
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            grid[i+j*pattern[0],0] = 0.2*pattern[1] - 0.2*(j) - 0.2
            grid[i+j*pattern[0],1] = 0.2*(i)
    grid2 = np.zeros(grid.shape)
    grid2[:,0] = grid[:,1]
    grid2[:,1] = grid[:,0]
    grid=grid2
    patches = []
    colors = []
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    for i in range(pattern[0]*pattern[1]):
        colors.append([weights[i],weights[i],weights[i]])
        rect = mpatches.Rectangle(grid[i], 0.2, 0.2, fill=True, facecolor=colors[i])
        ax.add_patch(rect)
        patches.append(rect)
        #ax.annotate(str(float_formatter(colors[i][0])), xy=(grid[i]+(0.005,0.072)), xytext=((grid[i]+(0.0052,0.0722))),fontsize=(180/pattern[0]),fontweight='bold')
        #ax.annotate(str(float_formatter(colors[i][1])), xy=(grid[i]+(0.005,0.042)), xytext=((grid[i]+(0.0052,0.0422))),fontsize=(180/pattern[0]),fontweight='bold')
        ax.annotate(str(float_formatter(colors[i][2])), xy=(grid[i]+(0.005,0.012)), xytext=((grid[i]+(0.0052,0.0122))),fontsize=(160/pattern[0]),fontweight='bold',color='g')
    #plt.subplots_adjust(left=0, right=0.2*pattern[0], bottom=0, top=0.2*pattern[1],wspace=0, hspace=0)
    #plt.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5,wspace=0, hspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    ax.set_ylim([0,0.2*pattern[0]])
    ax.set_xlim([0,0.2*pattern[1]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #plt.show()
    plt.savefig(str(pattern)+'_WithText_0o2.eps')
    plt.savefig(str(pattern)+'_WithText_0o2.svg')
    plt.gcf().clear()
    return