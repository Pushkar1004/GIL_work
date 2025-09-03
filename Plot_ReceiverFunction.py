import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seispy 
import matplotlib.ticker as mticker

class RFProfileData:
    """
    A class to bin, stack, and visualize receiver functions (RFs) along a profile.
    
    Parameters
    
    data : list of lists of numpy.ndarray
        Each inner list contains receiver functions (numpy arrays) for one station.
    stations : list of str
        Station names; i-th station corresponds to i-th list in "data".
    evloc : list of lists of tuple(float, float)
        Event locations (latitude, longitude); evloc[i][j] is the location of the j-th event for i-th station.
    staloc : list of tuple(float, float)
        Station locations (latitude, longitude); one tuple per station.
    
    Optional Parameters
    
    evmag : list of lists of float, default=None
        Event magnitudes; same structure as evloc. If None, magnitude filtering is disabled.
    snr : list of lists of float, default=None
        Signal-to-noise ratios; same structure as evloc. If None, SNR filtering is disabled.
    profile_stations : list of str, default=None
        Subset of stations to include in profile. If None, all stations are included.
    shift : float, default=None
        Time (s) before P-arrival. If provided with `time_after`, defines the time axis as [-shift, time_after].
    time_after : float, default=None
        Time (s) after P-arrival. Used with `shift` to define time axis. If not provided, time axis is [0, nsamples].
    nsamples : int, default=None
        Number of samples in each RF trace. If None, inferred from data (len of first RF trace).
        Note each rf should have same length. If not define the nsamples and stations that have same nsamples.
    """
    
    def __init__(self, data, stations, evloc, staloc, shift = None, time_after= None, 
                 nsamples = None, evmag = None, snr = None , profile_stations = None):
        
        self.data = data
        self.stations = stations
        self.evloc = evloc
        self.staloc = staloc
        self.shift = shift
        self.time_after = time_after
        self.nsamples = nsamples if nsamples is not None else len(self.data[0][0])
        self.profile_stations = profile_stations if profile_stations is not None else stations
        self.bins = defaultdict(list)   # dict to hold binned RFs: key=(station, (epi_bin, baz_bin))
        self.sta_end = {}               # dict to hold last bin index for each station
        
        if shift is None:
            shift = 0
        if time_after is None:
            time_after = self.nsamples
        self.shift = shift
        self.time_after = time_after
        self.time = np.linspace(-self.shift, self.time_after, self.nsamples)

        
        # Optional args
        self.evmag = evmag
        self.snr = snr

    def _bin_and_stack(self, mag_thre=4.5, snr_thre=0.0,
                       epi_range=(30, 10, 100), baz_range=(0, 20, 360)):
        """
        Bin receiver functions for each station by epicentral distance and back azimuth.

        Parameters
        mag_thre : float, default=4.5
            Events with smaller magnitude are discarde.
        snr_thre : float, default=0.0
            RFs below this SNR are discarded.
        epi_range: tuple (start, bin_size, end) for epicentral distance bins
            e.g. (30, 10, 100) → bins: 30–40, 40–50, ... up to 100
        baz_range: tuple (start, bin_size, end) for back azimuth bins
            e.g. (0, 20, 360) → bins: 0–20, 20–40, ... up to 360

        Returns
        self.bins : dict
            Dictionary with bin_info as key = "station_(baz_bin)_(epi_bin)", values = list of RFs in that bin.
            This can be used for Redatuming 
        """
        epi_bins = np.arange(epi_range[0], epi_range[2] + epi_range[1], epi_range[1])
        baz_bins = np.arange(baz_range[0], baz_range[2] + baz_range[1], baz_range[1])
        self.bins = defaultdict(list)
        self.sta_end = {}

        # create all keys(dtype string) across the stations in format station_baz_epi  
        for sta in self.stations:
            for baz in baz_bins:    
                for epi in epi_bins:
                    key = f"{sta}_{baz}_{epi}"
                    self.bins[key] = []

        for sta in self.profile_stations:
            if sta in self.stations:
                i = self.stations.index(sta)
                sta_la, sta_lo = self.staloc[i]

                for j, rf in enumerate(self.data[i]):
                    # Normalize RF to zero mean and unit variance
                    # rf = (rf - np.mean(rf)) / np.std(rf)
                    ev_la, ev_lo = self.evloc[i][j]
                    da = seispy.distaz(sta_la, sta_lo, ev_la, ev_lo)
                    dis, baz = da.delta, da.baz
                    
                    # Default values: +infinity so that if snr and mag does not exist we consider all data to plot    
                    if self.snr is None:
                        snr_val = np.inf
                    else:
                        snr_val = self.snr[i][j]

                    if self.evmag is None:
                        mag_val = np.inf
                    else:
                        mag_val = self.evmag[i][j]

                        
                    # Apply thresholds
                    if (mag_val < mag_thre or snr_val < snr_thre or
                        not (epi_range[0] < dis < epi_range[2]) or
                        not (baz_range[0] < baz < baz_range[2])):
                        continue

                    # Assign to bin
                    epi_bin = epi_bins[np.digitize(dis, epi_bins) - 1]  
                    baz_bin = baz_bins[np.digitize(baz, baz_bins) - 1]   

                    # Construct key
                    key = f"{sta}_{baz_bin}_{epi_bin}"

                    self.bins[key].append(rf)

        # Remove empty bins from the dictionary 
        self.bins = {k: v for k, v in self.bins.items() if v} 
        
        # Track where each station's bins end (for stacked plotting later)
        for i, key in enumerate(self.bins, start=0):
            self.sta_end[key.split("_")[0]] = i
        return self.bins 

    def _plot_all_stacked(self, scaling=1.0, spacing=5, figsize=(30, 20)):
        """
        Plot stacked RFs for each station along the profile.

        Notes
          RFs are stacked across bins for each station.
          Normalization: zero mean and unit variance.
        """
        yticks, yticklabels = [], []
        pre_sta_end = 0
        fig, ax = plt.subplots(figsize=figsize)

        for i, sta in enumerate(self.sta_end):
            # Concatenate RFs for this station across bins
            sta_data = list(self.bins.values())[pre_sta_end:self.sta_end[sta]]
            sta_data = np.vstack(sta_data)
            sta_stacked_data = np.mean(sta_data, axis=0) if len(sta_data) >= 2 else sta_data[0]

            # Normalize stacked RF
            sta_stacked_data = (sta_stacked_data - np.mean(sta_stacked_data)) / np.std(sta_stacked_data)

            # Vertical offset for plotting
            offset_data = sta_stacked_data*scaling + i * spacing
            zero_line = np.ones(offset_data.shape) * i * spacing

            # Fill positive and negative lobes
            ax.fill_between(self.time, offset_data, zero_line,
                            where=offset_data > zero_line, color='red')
            ax.fill_between(self.time, offset_data, zero_line,
                            where=offset_data <= zero_line, color='blue')

            # Plot trace outline
            ax.plot(self.time, offset_data, color='black', linewidth=0.5)

            yticks.append(i * spacing)
            yticklabels.append(sta)
            pre_sta_end = self.sta_end[sta]

        ax.set_title("Stacked Receiver Functions by Station")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))  
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Station")
        fig.tight_layout()
        plt.close(fig)
        return fig, ax

    def _plot_binned_stacked(self, scaling=3, spacing=2, figsize=(30, 20)):
        """
        Plot binned & stacked RFs for each (station, bin).
        RFs are stacked within each bin defined by (epi_bin, baz_bin).
        """
        fig, ax = plt.subplots(figsize=figsize)

        for bin_i, (key, bin_data) in enumerate(self.bins.items(), start=1):
            # Stack RFs in the bin
            if not bin_data:   # empty list → skip
                continue
            bin_stacked = np.mean(bin_data, axis=0) if len(bin_data) >= 2 else bin_data[0]
            bin_stacked = (bin_stacked - np.mean(bin_stacked) ) / np.std(bin_stacked)

            # Apply scaling and horizontal offset
            offset_rf = bin_stacked * scaling + bin_i * spacing
            zero_line_rf = np.ones(offset_rf.shape) * bin_i * spacing

            # Fill positive and negative parts
            ax.fill_betweenx(self.time, zero_line_rf, offset_rf,
                             where=offset_rf > zero_line_rf, color='red')
            ax.fill_betweenx(self.time, zero_line_rf, offset_rf,
                             where=offset_rf <= zero_line_rf, color='blue')

            # Plot trace outline
            ax.plot(offset_rf, self.time, color="black", linewidth=0.5)

        ax.xaxis.set_ticks_position('top')
        ax.set_ylabel("Time (s)", fontsize=24)
        ax.set_xlabel("Station", fontsize=24)
        # ax.set_yticks(np.arange(-self.shift, self.time_after, 10))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax.set_xticks([p * spacing for p in self.sta_end.values()])
        ax.set_xticklabels(self.sta_end.keys(), fontsize=24)
        ax.invert_yaxis()
        plt.close(fig)
        return fig, ax
        """
        Display figures
        fig.patch.set_facecolor("white")  # To ensure axis labels are visible
        display(fig)
        """;
        



