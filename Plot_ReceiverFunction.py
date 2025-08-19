import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seispy 

def PlotSectionBinnedRF(loaded_data, stations, mag_thre = 4.5, snr_thre = 5,epi_range = (30,10,100), baz_range = (0,20,360)):
    
    """
    Plot receiver functions along a profile binned by epicentral distance and back azimuth.

    Parameters
    
    loaded_data : dict
        {
          "data": list of lists where each inner list contains receiver functions (numpy arrays) for one station,
          "station": list of station names (i-th station corresponds to i-th list in "data"),
          "evloc": list of lists of tuples (evlat, evlong); evloc[i][j] is location of j-th event for i-th station,
          "evmag": list of lists of floats; magnitude of events, same structure as evloc,
          "staloc": list of tuples (sta_lat, sta_long), one tuple for each station,
          "snr": list of lists of signal-to-noise ratios, same structure as evmag
        }

    stations : list
        Station names along the profile to be plotted
    mag_thre : float, default=4.5
        Only events with magnitude > mag_thre are considered.
    snr_thre : float, default=5
        Receiver functions calculated from seismograms with signal-to-noise ratio (SNR) 
        below snr_thre are discarded and not plotted.
    epi_range: tuple (start, bin_size, end) for epicentral distance bins
        e.g. (30, 10, 100) → bins: 30–40, 40–50, ... up to 100
    baz_range: tuple (start, bin_size, end) for back azimuth bins
        e.g. (0, 20, 360) → bins: 0–20, 20–40, ... up to 360

    Notes
    - Binning is done on (epi_bin, baz_bin).
    - This ensures nearby teleseismic events received at a station are grouped together.
    - All receiver functions should have identical length before calling this function.
    """
    profile_data,sta_end = [], {}
    epi_bins = np.arange(epi_range[0], epi_range[2] + epi_range[1], epi_range[1])
    baz_bins = np.arange(baz_range[0], baz_range[2] + baz_range[1], baz_range[1])

    for sta in stations:
        if sta in loaded_data["station"]:
            i = loaded_data["station"].index(sta)
            sta_la, sta_lo = loaded_data["staloc"][i][0], loaded_data["staloc"][i][1]
            sta_data_binned = defaultdict(list)
            for j,rf in enumerate(loaded_data["data"][i]):
                rf = (rf - np.mean(rf)) / np.std(rf)
                mag = loaded_data["evmag"][i][j]
                ev_la, ev_lo = loaded_data["evloc"][i][j][0], loaded_data["evloc"][i][j][1]  
                da = seispy.distaz(sta_la, sta_lo, ev_la, ev_lo) 
                dis, baz  = da.delta, da.baz
                snr = loaded_data["snr"][i][j]
                if (mag < mag_thre
                    or snr<snr_thre
                    or not (epi_range[0] < da.delta < epi_range[2])
                    or not (baz_range[0] < da.baz < baz_range[2])
                    ):
                    continue
                
                epi_bin = np.digitize(dis, epi_bins) - 1
                baz_bin = np.digitize(baz, baz_bins) - 1
                bin_key = (epi_bin, baz_bin)
                sta_data_binned[bin_key].append(rf)
            
            sta_binned_stacked = []
            for key in sorted(sta_data_binned.keys()):
                bin_data = sta_data_binned[key]
                if len(bin_data) >= 2:
                    bin_stacked_data = np.mean(bin_data, axis=0)
                else:
                    bin_stacked_data = bin_data[0]
                sta_binned_stacked.append(bin_stacked_data)
                    
            if len(sta_binned_stacked) > 0:
                profile_data.append(sta_binned_stacked)
                sta_end[sta] = sum(len(x) for x in profile_data)
        
    profile_data = np.vstack(profile_data)

    
    
    """
    Stacked all RFs for each station
    """
    
    yticks, yticklabels = [], []
    pre_sta_end,spacing = 0,5
    n , nt = profile_data.shape
    shift, time_after = 10, 40  # Should be changed as per the time window before and after P-arrival
    time = np.linspace(-shift, time_after, nt)   
    fig1, ax1 = plt.subplots()
    for i, sta in enumerate(sta_end):
        sta_data = profile_data[pre_sta_end:sta_end[sta]]
        if len(sta_data) >= 2:
            sta_stacked_data = np.mean(sta_data, axis=0)
        else:
            sta_stacked_data = sta_data[0]

        sta_stacked_data = (sta_stacked_data - np.mean(sta_stacked_data)) / np.std(sta_stacked_data)
        # Offset stacked data 
        offset_data = sta_stacked_data + i * spacing
        zero_line = np.ones(offset_data.shape) * i * spacing
        # Fill above mean
        ax1.fill_between(time, offset_data, zero_line, where=offset_data > i * spacing, color='red')
        # Fill below mean
        ax1.fill_between(time, offset_data, zero_line, where=offset_data <= i * spacing, color='blue')
        #add boundary line
        ax1.plot(time, offset_data, color='black', linewidth=0.5)

        yticks.append(i * spacing)
        yticklabels.append(sta)
        pre_sta_end = sta_end[sta]

    ax1.set_title("Stacked Trace(s)")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_xticks(np.arange(-shift, time_after, 10))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Station")
    fig1.tight_layout()
 
    """
    Binned plot 
    """
    positions, labels = list(sta_end.values()), list(sta_end.keys())
    print(n, nt)
    width = max(30, 0.1*n)   # can be changed for better visualization
    height = 0.01*nt         # can be changed for better visualization 
    fig2, ax2 = plt.subplots(figsize=(width, height))
    spacing = 2   # can be changed for better visualization
    if n < 100:
        scaling = 1  # can be changed for better visualization
    else:
        scaling = 3  # can be changed for better visualization

    for i, rf in enumerate(profile_data):
        offset_rf = rf*scaling + i * spacing
        zero_line_rf = np.ones(offset_rf.shape) * i * spacing

        # Fill above baseline
        ax2.fill_betweenx(time, zero_line_rf, offset_rf, where=offset_rf > zero_line_rf, color='red')
        # Fill below baseline
        ax2.fill_betweenx(time, zero_line_rf, offset_rf, where=offset_rf <= zero_line_rf, color='blue')
        # Plot waveform
        ax2.plot(offset_rf, time, color="black", linewidth=0.5)


    ax2.xaxis.set_ticks_position('top')
    ax2.set_ylabel("Time (s)", fontsize=24)
    ax2.set_xlabel("Station", fontsize=24)
    ax2.set_yticks(np.arange(-shift, time_after, 10))
    ax2.set_xticks([p * spacing for p in positions])
    ax2.set_xticklabels(labels, fontsize=24)
    ax2.invert_yaxis()
    # Prevent plots from displaying at function call
    plt.close(fig1)
    plt.close(fig2)

    # Return figure and axes for later use (e.g., saving or customizing)
    return (fig1, ax1), (fig2, ax2) 

"""
This function returns matplotlib figure and axis for plotting.

(fig1, ax1), (fig2, ax2) = PlotSectionBinnedRF(
    loaded_data,
    stations=profile_stations,
    mag_thre=mag_thre,
    snr_thre=snr_thre,
    epi_range=epi_range,
    baz_range=baz_range
)

# Display figures
fig1.patch.set_facecolor("white")  # To ensure axis labels are visible
fig2.patch.set_facecolor("white")
display(fig1)
display(fig2)

# Save figures
fig1.savefig("name.png")
"""
