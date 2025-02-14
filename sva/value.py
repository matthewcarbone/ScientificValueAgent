"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from attrs import define, field
from monty.json import MSONable
from scipy.spatial import distance_matrix


def distance_matrix_pbc_vectorized(coords, box_side_lengths):
    """
    Vectorized calculation of pairwise distances under periodic boundary conditions.

    Parameters:
        coords (numpy.ndarray): NxD array of N points in D-dimensional space.
        box_length (float): Length of the sides of the simulation box.

    Returns:
        numpy.ndarray: NxN matrix of pairwise distances.
    """
    coords = np.array(coords)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    box_side_lengths = box_side_lengths.reshape(1, 1, -1)
    assert box_side_lengths.shape[-1] == delta.shape[-1]
    delta = delta - np.rint(delta / box_side_lengths) * box_side_lengths
    dist_matrix = np.sqrt((delta**2).sum(axis=-1))
    return dist_matrix


def compute_distances(X, metric_type="euclidean", box_side_lengths=None, 
                     anisotropic_params=None):
    """
    Compute pairwise distances with support for different metrics and PBC.
    
    Parameters
    ----------
    X : (N, d) np.ndarray
        Input data points
    metric_type : str
        Type of distance metric to use ("euclidean" or "anisotropic")
    box_side_lengths : array_like, optional
        If not None, use periodic boundary conditions with these box dimensions
    anisotropic_params : dict, optional
        Parameters for anisotropic distance calculation:
        - k_neighbors: int, number of neighbors for local PCA
        - alpha: float, exponent for eigenvalue scaling
        
    Returns
    -------
    dist_matrix : (N, N) np.ndarray
        Matrix of pairwise distances
    """
    if metric_type == "euclidean":
        if box_side_lengths is not None:
            return distance_matrix_pbc_vectorized(X, box_side_lengths)
        return distance_matrix(X, X)
    
    elif metric_type == "anisotropic":
        if anisotropic_params is None:
            raise ValueError("anisotropic_params required for anisotropic metric")
            
        k = anisotropic_params.get('k_neighbors', 6)
        alpha = anisotropic_params.get('alpha', 0.5)
        
        # Compute base distances for k-NN (using PBC if specified)
        if box_side_lengths is not None:
            base_dists = distance_matrix_pbc_vectorized(X, box_side_lengths)
        else:
            base_dists = distance_matrix(X, X)
            
        # Compute metric tensors using the base distances
        M_list = compute_local_metric(X, k=k, alpha=alpha, 
                                    base_distances=base_dists,
                                    box_side_lengths=box_side_lengths)
        
        # Compute final distances with the metric tensors
        if box_side_lengths is not None:
            return compute_anisotropic_distances_pbc(X, M_list, box_side_lengths)
        
        # Non-PBC case
        N = X.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            diff = X - X[i]  # (N, d)
            dist_matrix[i] = np.sqrt(np.sum(diff @ M_list[i] * diff, axis=1))
            
        return dist_matrix
    
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def compute_local_metric(X, k=6, alpha=0.5, eps=1e-6, base_distances=None,
                        box_side_lengths=None):
    """
    Compute anisotropic metric tensors M_i for each data point using PCA on k-NN.
    
    Parameters
    ----------
    X : (N, d) np.ndarray
        Input data points
    k : int
        Number of nearest neighbors to use for local PCA
    alpha : float
        Exponent for eigenvalue scaling in M_i = U Lambda^(-alpha) U^T
    eps : float
        Small value for numerical stability
    base_distances : (N, N) np.ndarray, optional
        Pre-computed distance matrix for finding k-NN
    box_side_lengths : array_like, optional
        If not None, use PBC when computing local neighborhoods
        
    Returns
    -------
    M_list : list of (d, d) np.ndarray
        List of local metric tensors M_i for each data point
    """
    N, d = X.shape
    M_list = []
    
    # Use provided distances or compute them
    if base_distances is None:
        if box_side_lengths is not None:
            dist_mat = distance_matrix_pbc_vectorized(X, box_side_lengths)
        else:
            dist_mat = distance_matrix(X, X)
    else:
        dist_mat = base_distances
    
    for i in range(N):
        # Find k-nearest neighbors
        knn_idx = np.argsort(dist_mat[i])[:k+1]  # Include self
        neighbors = X[knn_idx]
        
        if box_side_lengths is not None:
            # Apply PBC to neighbor positions relative to center point
            diff = neighbors - X[i]
            diff = diff - np.rint(diff / box_side_lengths) * box_side_lengths
            neighbors = X[i] + diff
        
        # Center the neighbors
        centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute local covariance
        C_i = centered.T @ centered / (len(neighbors) - 1)
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C_i)
        
        # Ensure positive eigenvalues and apply scaling
        eigvals = np.maximum(eigvals, eps)
        Lambda_inv = np.diag(1.0 / (eigvals**alpha))
        
        # Construct M_i
        M_i = eigvecs @ Lambda_inv @ eigvecs.T
        M_list.append(M_i)
    
    return M_list

def compute_anisotropic_distances(X, M_list):
    """
    Compute pairwise distances using local metric tensors.
    
    Parameters
    ----------
    X : (N, d) np.ndarray
        Input data points.
    M_list : list of (d, d) np.ndarray
        List of local metric tensors M_i for each data point.
        
    Returns
    -------
    dist_matrix : (N, N) np.ndarray
        Matrix of pairwise distances under local metrics.
    """
    N = X.shape[0]
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        diff = X - X[i]  # (N, d)
        dist_matrix[i] = np.sqrt(np.sum(diff @ M_list[i] * diff, axis=1))
    
    return dist_matrix

def compute_anisotropic_distances_pbc(X, M_list, box_side_lengths):
    """
    Compute anisotropic distances with periodic boundary conditions.
    
    Parameters
    ----------
    X : (N, d) np.ndarray
        Input data points
    M_list : list of (d, d) np.ndarray
        List of local metric tensors M_i for each point
    box_side_lengths : array_like
        Size of each dimension for PBC
        
    Returns
    -------
    dist_matrix : (N, N) np.ndarray
        Matrix of pairwise anisotropic distances with PBC
    """
    N = X.shape[0]
    box_side_lengths = np.asarray(box_side_lengths).reshape(1, 1, -1)
    
    # Compute all pairwise differences at once
    delta = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # Shape: (N, N, d)
    
    # Apply PBC to differences
    delta = delta - np.rint(delta / box_side_lengths) * box_side_lengths
    
    # Initialize distance matrix
    dist_matrix = np.zeros((N, N))
    
    # For each point i, compute distances using its metric tensor M_i
    for i in range(N):
        # Transform differences using M_i
        # (N, d) @ (d, d) -> (N, d)
        transformed = delta[:, i] @ M_list[i]  
        # Compute distances using transformed differences
        dist_matrix[i] = np.sqrt(np.sum(transformed * delta[:, i], axis=1))
    
    return dist_matrix

def svf(
    X,
    Y,
    sd=None,
    multiplier=1.0,
    characteristic_length="min",  # can now be "k_neighbors_mean"
    k=2,  # used if characteristic_length="k_neighbors_mean"
    density=False,
    gamma=1.0,  # fractional exponent for density normalization
    symmetric=False,
    scale=True,
    square_exponent=False,
    denominator_pbc=False,
    box_side_lengths=None,
    eps=1e-9,
    anisotropic=False,  # New parameter
    aniso_k_neighbors=6,  # New parameter
    aniso_alpha=0.5,  # New parameter
):
    """
    The value of two datasets, X and Y. Both X and Y must have the same
    number of rows. The returned result is an array of SVF values for each
    data point.

    Parameters
    ----------
    X : (N, d) np.ndarray
        The input data of shape N x d.
    Y : (N, d') np.ndarray
        The output (observations) data of shape N x d'.
    sd : float or np.ndarray, optional
        If provided, overrides automatic determination of length scale(s).
        If None, we use 'characteristic_length' to compute an sd per-point.
    multiplier : float, optional
        Multiplies the automatically derived length scale if 'sd' is None.
    characteristic_length : {"min", "max", "mean", "median", "global_mean", "k_neighbors_mean", "k_neighbors_min"}
        The operation to get the characteristic length if 'sd' is None.
        - "min": nearest neighbor distance
        - "max": maximum distance
        - "mean": average distance per point
        - "median": median distance per point
        - "global_mean": single mean value across all distances
        - "k_neighbors_mean": average of the first k nonzero nearest neighbors
        - "k_neighbors_min": k-th nearest neighbor distance (k=1 gives same as "min")
    k : int, optional
        Used if characteristic_length="k_neighbors_mean". 
        We take the average of the first k nearest (non-identical) distances.
    density : bool, optional
        If True, apply density-based normalization on the computed SVF.
    gamma : float, optional
        Exponent for fractional density normalization. 
        - If gamma=1.0, this is the classic full normalization. 
        - 0 < gamma < 1 yields partial normalization.
    symmetric : bool, optional
        If True, uses ls_ij = sqrt(sd[i]*sd[j]) instead of ls_ij = sd[i].
    scale : bool, optional
        If True, rescale the final SVF values to [0, 1].
    square_exponent : bool, optional
        If True, uses exp(- (dist^2) / sd^2), else exp(- dist / sd).
    denominator_pbc : bool, optional
        If True, compute the density denominator with periodic boundary conditions.
        Requires 'box_side_lengths'.
    box_side_lengths : array_like, optional
        If denominator_pbc=True, the size of each dimension for PBC.
    eps : float, optional
        A small offset to avoid division by zero when normalizing.
    anisotropic : bool, optional
        If True, use anisotropic distance metrics based on local PCA.
    aniso_k_neighbors : int, optional
        Number of neighbors to use for local PCA when anisotropic=True.
    aniso_alpha : float, optional
        Exponent for eigenvalue scaling in anisotropic metric tensor.

    Returns
    -------
    v : (N,) np.ndarray
        The SVF value for each data point.
    """
    # --------------------------------------------------
    # 1) Compute distance matrices
    # --------------------------------------------------
    if anisotropic:
        aniso_params = {
            'k_neighbors': aniso_k_neighbors,
            'alpha': aniso_alpha
        }
        X_dist = compute_distances(X, metric_type="anisotropic",
                                 box_side_lengths=box_side_lengths if denominator_pbc else None,
                                 anisotropic_params=aniso_params)
    else:
        X_dist = compute_distances(X, metric_type="euclidean",
                                 box_side_lengths=box_side_lengths if denominator_pbc else None)
    
    Y_dist = distance_matrix(Y, Y)

    # --------------------------------------------------
    # 2) Compute sd (characteristic length) if None
    # --------------------------------------------------
    if sd is None:
        # Make a local copy to modify
        dist_copy = X_dist.copy()
        # Replace exact zeros with NaN to exclude self-distances
        dist_copy[np.isclose(dist_copy, 0.0)] = np.nan

        if characteristic_length == "min":
            # nearest-neighbor distance
            sd_vals = np.nanmin(dist_copy, axis=1) * multiplier

        elif characteristic_length == "max":
            # maximum distance to any other point
            sd_vals = np.nanmax(dist_copy, axis=1) * multiplier

        elif characteristic_length == "mean":
            # mean distance
            sd_vals = np.nanmean(dist_copy, axis=1) * multiplier

        elif characteristic_length == "median":
            # median distance
            sd_vals = np.nanmedian(dist_copy, axis=1) * multiplier

        elif characteristic_length == "global_mean":
            # single mean value across all distances (excluding self-distances)
            global_mean = np.nanmean(dist_copy) * multiplier
            sd_vals = np.full(dist_copy.shape[0], global_mean)

        elif characteristic_length == "k_neighbors_mean":
            # average over the first k non-NaN distances
            # We'll sort each row and then average the first k valid distances
            N = dist_copy.shape[0]
            sd_list = []
            for i in range(N):
                row = dist_copy[i, :]
                valid_dists = np.sort(row[~np.isnan(row)])
                if len(valid_dists) == 0:
                    # Degenerate case: no valid distances?
                    # fallback to 1.0 or some default
                    sd_list.append(1.0 * multiplier)
                else:
                    # If fewer than k distances, average all we have
                    cutoff = min(len(valid_dists), k)
                    sd_list.append(valid_dists[:cutoff].mean() * multiplier)
            sd_vals = np.array(sd_list)

        elif characteristic_length == "k_neighbors_min":
            # k-th nearest neighbor distance (k=1 gives same as "min")
            N = dist_copy.shape[0]
            sd_list = []
            for i in range(N):
                row = dist_copy[i, :]
                valid_dists = np.sort(row[~np.isnan(row)])
                if len(valid_dists) == 0:
                    # Degenerate case: no valid distances?
                    # fallback to 1.0 or some default
                    sd_list.append(1.0 * multiplier)
                else:
                    # If fewer than k distances, take the last available one
                    k_idx = min(k - 1, len(valid_dists) - 1)  # k=1 means index 0
                    sd_list.append(valid_dists[k_idx] * multiplier)
            sd_vals = np.array(sd_list)

        else:
            raise ValueError(f"Unknown characteristic_length '{characteristic_length}'")

        sd = sd_vals.reshape(1, -1)  # shape (1, N)

    # --------------------------------------------------
    # 3) Construct the local length scale matrix
    # --------------------------------------------------
    if symmetric:
        # Symmetric => ls_ij = sqrt(sd[i]*sd[j])
        ls = np.sqrt(sd * sd.T)
    else:
        # Asymmetric => ls_ij = sd[i]
        ls = sd

    # --------------------------------------------------
    # 4) Compute the weighting function w(i,j)
    # --------------------------------------------------
    if square_exponent:
        arg = -(X_dist**2) / (ls**2)
    else:
        arg = -X_dist / ls

    w = np.exp(arg)  # (N,N) pairwise weights
    numerator = Y_dist * w  # numerator for the SVF

    # --------------------------------------------------
    # 5) Compute final SVF
    # --------------------------------------------------
    if not density:
        # No density normalization => just average the numerator
        v_vals = numerator.mean(axis=1)
    else:
        # We do density normalization
        if denominator_pbc:
            if box_side_lengths is None:
                raise ValueError("`box_side_lengths` must be provided if denominator_pbc=True")
            # Recompute distances for the denominator using PBC
            X_dist_pbc = distance_matrix_pbc_vectorized(X, box_side_lengths)
            if square_exponent:
                arg_pbc = -(X_dist_pbc**2) / (ls**2)
            else:
                arg_pbc = -X_dist_pbc / ls
            w_pbc = np.exp(arg_pbc)
            denominator = w_pbc.mean(axis=1)
        else:
            denominator = w.mean(axis=1)

        # Fractional density normalization
        # v_i = mean_j(numerator) / ((mean_j(w))^gamma + eps)
        v_vals = numerator.mean(axis=1) / ((denominator**gamma) + eps)

    # --------------------------------------------------
    # 6) Optionally rescale v to [0, 1]
    # --------------------------------------------------
    if scale:
        vmin = v_vals.min()
        vmax = v_vals.max()
        if np.isclose(vmin, vmax):
            # degenerate case => return zeros
            return np.zeros_like(v_vals)
        # Min-max scale
        v_scaled = (v_vals - vmin) / (vmax - vmin)
        return v_scaled
    else:
        return v_vals



@define
class SVF(MSONable):
    sd = field(default=None)
    multiplier = field(default=1.0)
    characteristic_length = field(default="min")
    density = field(default=False)
    symmetric = field(default=False)
    scale = field(default=True)
    square_exponent = field(default=False)
    denominator_pbc = field(default=False)
    box_side_lengths = field(default=None)
    anisotropic = field(default=False)
    aniso_k_neighbors = field(default=6)
    aniso_alpha = field(default=0.5)

    def __call__(self, X, Y):
        return svf(
            X,
            Y,
            sd=self.sd,
            multiplier=self.multiplier,
            characteristic_length=self.characteristic_length,
            density=self.density,
            symmetric=self.symmetric,
            scale=self.scale,
            square_exponent=self.square_exponent,
            denominator_pbc=self.denominator_pbc,
            box_side_lengths=self.box_side_lengths,
            anisotropic=self.anisotropic,
            aniso_k_neighbors=self.aniso_k_neighbors,
            aniso_alpha=self.aniso_alpha,
        )
