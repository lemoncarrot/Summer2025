from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.mixture import GaussianMixture
from pomegranate import *
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import Gamma
from skimage.filters import threshold_otsu, threshold_triangle

# function to add annotations by gaussian mixture model (GMM) stratification to obs labels

def stratify_by_marker(ann_data_obj, marker_name, gmm_components):
    log_exprs = ann_data_obj[:, marker_name].layers["log1p"].flatten()

    gmm = GaussianMixture(n_components=gmm_components, random_state=0)
    gmm.fit(log_exprs.reshape(-1, 1))

    raw_labels = gmm.predict(log_exprs.reshape(-1, 1))

    # ensure that higher expression group gets marked as 1
    means = gmm.means_.flatten()
    order = np.argsort(means)
    remapped_labels = np.array([np.where(order==lbl)[0][0] for lbl in raw_labels])

    ann_data_obj.obs[f"{marker_name}_strat"] = remapped_labels

# trying a method adapted from GammaGateR paper
# first transforming the data so the "spike" shifts to around zero 
# trans = max(epsilon, x - some percentile)

# modeling using a two-component gamma mixture model
def gamma_strat(ann_data_obj, marker_name, components, p_thresh=0.9):
    raw_exprs = ann_data_obj[:, marker_name].layers["raw"].flatten()
    first_percentile = np.quantile(raw_exprs, 0.01)
    epsilon = 1e-5

    transformed_exprs = np.maximum(epsilon, raw_exprs - first_percentile).reshape(-1, 1)

    # do gamma mixture model
    model = GeneralMixtureModel([Gamma(), Gamma()] if components == 2
                                    else [Gamma() for _ in range(components)],
                                    random_state=0, verbose=False)
    model.fit(transformed_exprs)
    
    #params = [p.detach().cpu().numpy() for p in model.distributions[0].parameters()]
    #print(params)
    #print("line has run")
    
    means = []
    for d in model.distributions:
        params = [p.detach().cpu().numpy() for p in d.parameters()]
        #print(params)
        loc, shape, rate = [float(p) for p in params]
        means.append(loc + (shape / rate))
    #print("loop done running")
    pos_idx = int(np.argmax(means))

    # posteriors
    R = model.predict_proba(transformed_exprs).detach().cpu().numpy()             # (N, K), rows sum to 1
    p_pos = R[:, pos_idx]                   # P(positive | x)
    labs = (p_pos >= p_thresh).astype(int)

    ann_data_obj.obs[f"{marker_name}_strat"] = labs

def otsu_thresholding(ann_data_obj, marker_name):
    temp = ann_data_obj[:, marker_name].layers["log1p"].flatten()
    thr = threshold_otsu(temp)
    labs = (temp >= thr).astype(int)

    ann_data_obj.obs[f"{marker_name}_strat"] = labs

def triangle_method(ann_data_obj, marker_name):
    temp = ann_data_obj[:, marker_name].layers["log1p"].flatten()
    thr = threshold_triangle(temp)
    labs = (temp >= thr).astype(int)

    ann_data_obj.obs[f"{marker_name}_strat"] = labs

# function to display bar plot by marker stratification

def plot_strat(df, ann_data_obj, marker_name):
    plt.figure(figsize=(10, 6))
    df[f"{marker_name}_strat"] = ann_data_obj.obs[f"{marker_name}_strat"]
    df.groupby("leiden_clusters")[f"{marker_name}_strat"].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True)
    plt.title(f"{marker_name} Stratification")
    plt.xlabel("Clusters")
    plt.ylabel("Prop")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# function to print for each cluster, how many cells are marked as positive for a given marker
# add specifications for if input is list of markers, and if the specification is OR or AND

def print_counts(df, num_clusters, markers_and_operator, print=False) -> list:
    markers = markers_and_operator[0]
    operator = markers_and_operator[1] if len(markers_and_operator) > 1 else "NONE"
    res = [0] * num_clusters
    for i in range(0, num_clusters):
        #if isinstance(markers, list):
        ident = [m + "_strat" for m in markers] # match to column names in adata.obs
        if operator == "OR":
            # count the cells that are positive (have a 1 in the corresponding column of adata.obs) for any of the markers in the list
            count = df[df[ident].any(axis=1) & (df["leiden_clusters"].astype(int) == i)].shape[0]
        elif operator == "AND" or operator == "NONE":
            # count cells that are positive for all markers in the list
            count = df[df[ident].all(axis=1) & (df["leiden_clusters"].astype(int) == i)].shape[0]
        else:
            raise ValueError("Something isn't right...")
        #else: # if there is only one marker in the query
            #ident = markers + "_strat" # match to column names in adata.obs
            #count = df[(df[ident] == 1) & (df["leiden_clusters"].astype(int) == i)].shape[0]
        if print:
            print(f"Cluster {i}: {count} cells marked as positive for {markers} with operator {operator} *operator not applicable if marker is a single value")
        res[i] = count


    return res