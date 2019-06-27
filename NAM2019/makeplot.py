def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]    
    return [(int(i[:2], 16)/256., int(i[2:4], 16)/256., int(i[4:], 16)/256.) for i in colors]

def gencmapcolors(num_colors, cmap = 'tab20', shuffle=False):
    import pylab, random
    cm = pylab.get_cmap(cmap)
    colors = [cm(1.*i/(num_colors*1.0)) for i in range(num_colors)]
    if (shuffle):
        random.shuffle(colors)
    return colors

x = yeararr[11:]
npapers = yearsum[11:]-yearhist['other'][11:] # Total number of papers in year
colors = dict(zip(abstexts,gencmapcolors(len(abstexts))))

baseperc = zeros(len(x))
for abstext in abstexts:
    mpapers = yearhist[abstext][11:] # Number of those for given abstext
    perc    = mpapers   
    bar(x, perc, bottom=baseperc,edgecolor='white',color=colors[abstext],label=abstext)
    baseperc=baseperc+perc

#legend()
#barWidth = 0.85

# Custom x axis
plt.xlabel("Year",fontsize=16)
plt.ylabel("Number",fontsize=16)

# Force xticks to integers
plt.xticks(rotation=90,)
from matplotlib.ticker import MaxNLocator
ax = gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
tight_layout()

savefig("ML_numbers.png")

figure()
baseperc = zeros(len(x))
for abstext in abstexts:
    mpapers = yearhist[abstext][11:] # Number of those for given abstext
    perc    = mpapers/npapers    
    bar(x, perc, bottom=baseperc,edgecolor='white',color=colors[abstext],label=abstext)
    baseperc=baseperc+perc

#legend()
#barWidth = 0.85

# Custom x axis
plt.xlabel("Year",fontsize=16)
plt.ylabel("Fraction",fontsize=16)

# Force xticks to integers
plt.xticks(rotation=90,)
from matplotlib.ticker import MaxNLocator
ax = gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
tight_layout()

savefig("ML_frac.png")

