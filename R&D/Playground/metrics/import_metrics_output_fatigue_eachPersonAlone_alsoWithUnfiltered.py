"""
Import the exported metrics from multiple JSON files,
produce the 4 fatigue indeces from each selected person,
lowpass filter each person alone,
plot each person alone.
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal




avg_firsts_for_filter_size = 10

def main():

    filepaths = open_dialog_and_select_multiple_files()
    for filepath in filepaths:
        with open(filepath, "r") as json_file:
            m = json.load(json_file)

        m['mnf_arv_ratio'] = -np.array(m['mnf_arv_ratio'])
        m['emd_mdf1'] = -np.array(m['emd_mdf1'])
        m['emd_mdf2'] = -np.array(m['emd_mdf2'])
        m.pop('person')

        plot_with_unfiltered = True

        fs = 4
        cutoff = 0.08  # Cutoff frequency in Hz
        order = 2    # Filter order
        normal_cutoff = cutoff/ (fs / 2)
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        

        baseMetrics = {
            'rms': m['rms'][0],
            "mnf_arv_ratio": m['mnf_arv_ratio'][0],
            "ima_diff": m['ima_diff'][0],
            "emd_mdf1": m['emd_mdf1'][0],
            "emd_mdf2": m['emd_mdf2'][0],
            "fluct_variance": m['fluct_variance'][0],
            "fluct_range_values": m['fluct_range_values'][0],
            "fluct_mean_diff_values": m['fluct_mean_diff_values'][0]
        }


        # fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        # plt.title(f"{filepaths[0].split('/')[-1].split('_ID')[0]}")
        # axs[0, 0].plot(m["rms"], label="rms")
        # axs[0, 0].legend()
        # axs[0, 0].grid()
        
        # axs[0, 1].plot(m["emd_mdf1"], label="emd_mdf1")
        # axs[0, 1].plot(m["emd_mdf2"], label="emd_mdf2")
        # axs[0, 1].legend()
        # axs[0, 1].grid()
        
        # axs[1, 0].plot(m["mnf_arv_ratio"], label="mnf_arv_ratio")
        # axs[1, 0].legend()
        # axs[1, 0].grid()
        
        # axs[1, 1].plot(m["ima_diff"], label="ima_diff")
        # axs[1, 1].legend()
        # axs[1, 1].grid()
        
        # axs[2, 0].plot(m["fluct_variance"], label="fluct_variance")
        # axs[2, 0].plot(m["fluct_range_values"],   label="fluct_range_values")
        # axs[2, 0].plot(m["fluct_mean_diff_values"], label="fluct_mean_diff_values")
        # axs[2, 0].legend()
        # axs[2, 0].grid()

        # fig.delaxes(axs[2, 1])
        # plt.tight_layout()
        
        

        fig2, axs2 = plt.subplots(4, 1, figsize=(8, 9))
        plt.title(f"{filepath.split('/')[-1].split('_ID')[0]}")
        plt.tight_layout()

        weightedsum = signal.filtfilt(b, a, weighted_sum_fatigue(m))
        # plot_fatigue(fatigue1, "Weighted Sum Fatigue")


        calculator = FatigueIndexCalculator()
        fatigue_index = calculator.calculate_fatigue_index(m)
        contributions = calculator.get_metric_contribution(m)
        print(contributions)
        indexcalc = signal.filtfilt(b, a, fatigue_index)


        learner = FatigueLearner(m)
        pca = signal.filtfilt(b, a, learner.extract_fatigue_indicator(method='pca'))
        tsne = signal.filtfilt(b, a, learner.extract_fatigue_indicator(method='tsne'))

        if plot_with_unfiltered:
            axs2[0].plot(weighted_sum_fatigue(m))
            axs2[1].plot(fatigue_index)
            axs2[2].plot(learner.extract_fatigue_indicator(method='pca'))
            axs2[3].plot(learner.extract_fatigue_indicator(method='tsne'))

        axs2[0].plot(weightedsum, label='weightedsum')
        axs2[0].legend()
        axs2[0].grid()
        axs2[1].plot(indexcalc, label='indexcalc')
        axs2[1].legend()
        axs2[1].grid()
        axs2[2].plot(pca, label='pca')
        axs2[2].legend()
        axs2[2].grid()
        axs2[3].plot(tsne, label='tsne')
        axs2[3].legend()
        axs2[3].grid()

    plt.show()
    
    
    



#########################################################

class FatigueLearner:
    def __init__(self, metrics):
        """
        Initialize with metrics dictionary
        
        Args:
            metrics (dict): Dictionary of time series metrics
        """
        self.metrics_df = pd.DataFrame(metrics)
        self.scaler = StandardScaler()
        
    def pca_analysis(self, n_components=6):
        """
        Perform PCA to reduce dimensionality and find primary trends
        
        Args:
            n_components (int): Number of principal components to retain
        
        Returns:
            np.array: Transformed data
            np.array: Explained variance ratio
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.metrics_df)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        return pca_result, pca.explained_variance_ratio_
    
    def kmeans_clustering(self, n_clusters=20):
        """
        Perform K-means clustering to identify fatigue states
        
        Args:
            n_clusters (int): Number of fatigue states to identify
        
        Returns:
            np.array: Cluster labels
            np.array: Cluster centers
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.metrics_df)
        
        # Perform K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        
        return labels, kmeans.cluster_centers_
    
    def tsne_visualization(self, perplexity=5):
        """
        Use t-SNE for non-linear dimensionality reduction and visualization
        
        Args:
            perplexity (int): Controls balance between preservation of local 
                               and global structures
        
        Returns:
            np.array: 2D embedding of the data
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.metrics_df)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(scaled_data)
        
        return tsne_result
    
    def extract_fatigue_indicator(self, method, num=1):
        """
        Extract a fatigue indicator using different methods
        
        Args:
            method (str): 'pca', 'kmeans', or 'tsne'
        
        Returns:
            np.array: Fatigue indicator
        """
        if method == 'pca':
            pca_result, variance_ratio = self.pca_analysis(n_components=num)
            # Use first principal component as fatigue indicator
            fatigue = pca_result[:, 0]
            fatigue[0] = np.mean(fatigue[:avg_firsts_for_filter_size])
            return fatigue
        
        elif method == 'kmeans':
            labels, centers = self.kmeans_clustering()
            # Use cluster labels as fatigue stages
            return labels
        
        elif method == 'tsne':
            tsne_result = self.tsne_visualization()
            # Use first dimension of t-SNE as potential fatigue indicator
            fatigue = tsne_result[:, 0]
            fatigue[0] = np.mean(fatigue[:avg_firsts_for_filter_size])
            return fatigue
        
        else:
            raise ValueError("Invalid method. Choose 'pca', 'kmeans', or 'tsne'.")

def visualize_fatigue_indicator(fatigue_indicator, title, a, b):
    """
    Simple visualization of fatigue indicator
    
    Args:
        fatigue_indicator (np.array): Fatigue indicator values
    """
    plt.figure(figsize=(10, 4))
    plt.tight_layout()
    plt.plot(fatigue_indicator)
    signal_filtered = signal.filtfilt(b, a, fatigue_indicator)
    plt.plot(signal_filtered)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Fatigue Level')

class FatigueIndexCalculator:
    def __init__(self):
        # Define which metrics increase/decrease with fatigue
        self.increasing_metrics = {
            'rms': 1.0,           # Typically increases with fatigue
            'ima_diff': 1.0,      # Assuming increases with fatigue
            'fluct_variance': 1.0,
            'fluct_range_values': 1.0,
            'fluct_mean_diff_values': 1.0
        }
        
        self.decreasing_metrics = {
            'mnf_arv_ratio': 1.0, # Typically decreases with fatigue
            'emd_mdf1': 1.0,      # Frequency metrics typically decrease
            'emd_mdf2': 1.0
        }
        
    def normalize_metric(self, values, is_increasing):
        """
        Normalize metric values while preserving trend direction
        """
        if len(values) == 0:
            return []
            
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            return np.zeros_like(values)
            
        if is_increasing:
            return (values - min_val) / (max_val - min_val)
        else:
            return 1 - (values - min_val) / (max_val - min_val)
    
    def calculate_fatigue_index(self, metrics, window_size=None):
        """
        Calculate fatigue index from multiple metrics
        
        Args:
            metrics (dict): Dictionary of metric lists
            window_size (int): Optional sliding window size for smoothing
            
        Returns:
            numpy.array: Fatigue index over time
        """
        # Validate input
        if not all(len(v) == len(next(iter(metrics.values()))) for v in metrics.values()):
            raise ValueError("All metric lists must have the same length")
        
        if len(next(iter(metrics.values()))) == 0:
            return np.array([])
            
        normalized_metrics = {}
        
        # # Normalize increasing metrics
        # for metric_name in self.increasing_metrics:
        #     if metric_name in metrics:
        #         normalized_metrics[metric_name] = metrics[metric_name]
        #         # normalized_metrics[metric_name] = self.normalize_metric(
        #         #     metrics[metric_name], True
        #         # ) * self.increasing_metrics[metric_name]
        
        # # Normalize decreasing metrics
        # for metric_name in self.decreasing_metrics:
        #     if metric_name in metrics:
        #         normalized_metrics[metric_name] = -np.array(metrics[metric_name])
        #         # normalized_metrics[metric_name] = self.normalize_metric(
        #         #     metrics[metric_name], False
        #         # ) * self.decreasing_metrics[metric_name]
        
        # Combine all normalized metrics
        all_metrics = np.array(list(metrics.values()))
        fatigue_index = np.mean(all_metrics, axis=0)
        
        # Apply smoothing if window_size is specified
        if window_size:
            kernel = np.ones(window_size) / window_size
            fatigue_index = np.convolve(fatigue_index, kernel, mode='same')
        
        fatigue_index[0] = np.mean(fatigue_index[:avg_firsts_for_filter_size])
        return fatigue_index
    
    def set_metric_weights(self, weights_dict):
        """
        Update weights for specific metrics
        
        Args:
            weights_dict (dict): Dictionary of metric names and their weights
        """
        for metric, weight in weights_dict.items():
            if metric in self.increasing_metrics:
                self.increasing_metrics[metric] = weight
            elif metric in self.decreasing_metrics:
                self.decreasing_metrics[metric] = weight
                
    def get_metric_contribution(self, metrics):
        """
        Calculate contribution of each metric to final fatigue index
        
        Returns:
            dict: Contribution of each metric
        """
        contributions = {}
        
        for metric_name in self.increasing_metrics:
            if metric_name in metrics:
                contributions[metric_name] = self.normalize_metric(
                    metrics[metric_name], True
                ).mean() * self.increasing_metrics[metric_name]
                
        for metric_name in self.decreasing_metrics:
            if metric_name in metrics:
                contributions[metric_name] = self.normalize_metric(
                    metrics[metric_name], False
                ).mean() * self.decreasing_metrics[metric_name]
                
        return contributions

def weighted_sum_fatigue(m, weights=None):
    """Simple weighted sum method."""
    metrics = np.array(list(m.values()))  # Shape: (num_features, num_samples)
    if weights is None:
        weights = np.ones(metrics.shape[0])  # Equal weights if none provided
    fatigue = np.dot(weights, metrics)  # Weighted sum
    fatigue[0] = np.mean(fatigue[:avg_firsts_for_filter_size])
    return fatigue.tolist()

def kmeans_fatigue(m, n_clusters=100):
    """K-Means clustering method."""
    data = np.array(list(m.values())).T  # Shape: (num_samples, num_features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    return labels.tolist()

def gaussian_mixture_fatigue(m, n_components=100):
    """Gaussian Mixture Model clustering method."""
    data = np.array(list(m.values())).T
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(data_scaled)
    return labels.tolist()

def pca_threshold_fatigue(m, threshold=0.5):
    """PCA-based dimensionality reduction with thresholding."""
    data = np.array(list(m.values())).T
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=1)
    fatigue_scores = pca.fit_transform(data_scaled).flatten()
    return (fatigue_scores > threshold).astype(int).tolist()

def progressive_fatigue(m):
    """Progressive index assuming increasing fatigue over time."""
    num_samples = len(next(iter(m.values())))
    fatigue = np.linspace(0, 1, num_samples)  # Linear progression
    return fatigue.tolist()

def plot_fatigue(fatigue, title):
    plt.figure(figsize=(8, 4))
    plt.plot(fatigue, label=title)
    plt.xlabel("Time")
    plt.ylabel("Fatigue Level")
    plt.title(title)
    plt.legend()

def open_dialog_and_select_multiple_files():
    """
    Opens a file dialog allowing the user to select multiple files.

    Returns:
        list: A list of file paths selected by the user.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_paths = filedialog.askopenfilenames(title="Select Files")

    # Convert to a list and return
    return list(file_paths)



if  __name__ == "__main__":
    main()