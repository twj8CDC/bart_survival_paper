import numpy as np
import math
# import pymc as pm
# import pymc_bart as pmb
from pymc_bart.pgbart import PGBART
from numba import njit
import scipy as sp


# ==========================================
# 1. THE NUMBA ROULETTE WHEEL (SPARSITY)
# ==========================================
@njit(fastmath=True)
def sample_feature(alpha_vec):
    total_weight = np.sum(alpha_vec)
    r = np.random.uniform(0.0, total_weight)
    cumulative = 0.0
    for i in range(len(alpha_vec)):
        cumulative += alpha_vec[i]
        if r <= cumulative:
            return i
    return len(alpha_vec) - 1

@njit(fastmath=True)
def get_split_counts(F, num_features):
    counts = np.zeros(num_features, dtype=np.int32)
    m_trees = F.shape[0]
    max_nodes = F.shape[1]
    for m in range(m_trees):
        for i in range(max_nodes):
            feat = F[m, i]
            if feat != -1: counts[feat] += 1
    return counts

# ==========================================
# 2. THE MUTATIONS
# ==========================================
@njit(fastmath=True)
def get_active_leaves(F_tree, current_node_count):
    leaves = np.empty(current_node_count, dtype=np.int16)
    leaf_count = 0
    for i in range(current_node_count):
        if F_tree[i] == -1:
            leaves[leaf_count] = i
            leaf_count += 1
    return leaves[:leaf_count]

@njit(fastmath=True)
def propose_grow(tree_idx, X, F, T, L, R, V, D, node_counts, alpha_vec, leaf_prior_std):
    current_count = node_counts[tree_idx]
    leaves = get_active_leaves(F[tree_idx], current_count)
    if current_count >= F.shape[1] - 1: return False, -1 
        
    random_idx = np.random.randint(0, len(leaves))
    leaf_to_split = leaves[random_idx]
    
    # NEW: Sparsity-aware feature selection
    split_feature = sample_feature(alpha_vec)
    feat_min, feat_max = np.min(X[:, split_feature]), np.max(X[:, split_feature])
    split_value = np.random.uniform(feat_min, feat_max)
    
    left_child_idx, right_child_idx = current_count, current_count + 1
    
    F[tree_idx, leaf_to_split] = split_feature
    T[tree_idx, leaf_to_split] = split_value
    L[tree_idx, leaf_to_split] = left_child_idx
    R[tree_idx, leaf_to_split] = right_child_idx
    F[tree_idx, left_child_idx] = -1
    F[tree_idx, right_child_idx] = -1
    
    # NEW: Update the Depth pointer
    parent_depth = D[tree_idx, leaf_to_split]
    D[tree_idx, left_child_idx] = parent_depth + 1
    D[tree_idx, right_child_idx] = parent_depth + 1

    # NEW: Inherit parent's value and add noise to the new leaves!
    parent_val = V[tree_idx, leaf_to_split]
    # V[tree_idx, left_child_idx] = parent_val + np.random.normal(0.0, leaf_prior_std)
    # V[tree_idx, right_child_idx] = parent_val + np.random.normal(0.0, leaf_prior_std)
    V[tree_idx, left_child_idx] = np.random.normal(0.0, leaf_prior_std)
    V[tree_idx, right_child_idx] = np.random.normal(0.0, leaf_prior_std)
    
    
    node_counts[tree_idx] += 2
    return True, leaf_to_split

@njit(fastmath=True)
def get_prunable_nodes(F_tree, L_tree, R_tree, current_node_count):
    prunable = np.empty(current_node_count, dtype=np.int16)
    count = 0
    for i in range(current_node_count):
        if F_tree[i] != -1:
            left, right = L_tree[i], R_tree[i]
            if left != -1 and right != -1:
                if F_tree[left] == -1 and F_tree[right] == -1:
                    prunable[count] = i
                    count += 1
    return prunable[:count]

@njit(fastmath=True)
def propose_prune(tree_idx, F, L, R, V, node_counts, leaf_prior_std):
    current_count = node_counts[tree_idx]
    prunable_nodes = get_prunable_nodes(F[tree_idx], L[tree_idx], R[tree_idx], current_count)
    if len(prunable_nodes) == 0: return False, -1, -1
        
    random_idx = np.random.randint(0, len(prunable_nodes))
    node_to_prune = prunable_nodes[random_idx]
    old_feature = F[tree_idx, node_to_prune]
    
    F[tree_idx, node_to_prune] = -1
    V[tree_idx, node_to_prune] = np.random.normal(0.0, leaf_prior_std)
    return True, node_to_prune, old_feature

@njit(fastmath=True)
def get_internal_nodes(F_tree, current_node_count):
    internal = np.empty(current_node_count, dtype=np.int16)
    count = 0
    for i in range(current_node_count):
        if F_tree[i] != -1:
            internal[count] = i
            count += 1
    return internal[:count]

@njit(fastmath=True)
def propose_change(tree_idx, X, F, T, node_counts, alpha_vec):
    current_count = node_counts[tree_idx]
    internal_nodes = get_internal_nodes(F[tree_idx], current_count)
    if len(internal_nodes) == 0: return False, -1, -1, 0.0
        
    random_idx = np.random.randint(0, len(internal_nodes))
    node_to_change = internal_nodes[random_idx]
    
    old_feature = F[tree_idx, node_to_change]
    old_threshold = T[tree_idx, node_to_change]
    
    # NEW: Sparsity-aware feature selection
    new_feature = sample_feature(alpha_vec)
    feat_min, feat_max = np.min(X[:, new_feature]), np.max(X[:, new_feature])
    new_threshold = np.random.uniform(feat_min, feat_max)
    
    F[tree_idx, node_to_change] = new_feature
    T[tree_idx, node_to_change] = new_threshold
    
    return True, node_to_change, old_feature, old_threshold


@njit(fastmath=True)
def propose_leaf_change(tree_idx, F, V, node_counts, leaf_prior_std, step_size):
    current_count = node_counts[tree_idx]
    leaves = get_active_leaves(F[tree_idx], current_count)
    if len(leaves) == 0: return False, -1, 0.0, 0.0
    
    random_idx = np.random.randint(0, len(leaves))
    leaf_to_change = leaves[random_idx]
    
    old_val = V[tree_idx, leaf_to_change]
    new_val = old_val + np.random.normal(0.0, step_size)
    V[tree_idx, leaf_to_change] = new_val

    var = leaf_prior_std ** 2
    penalty = (-0.5 * (new_val ** 2) / var) - (-0.5 * (old_val ** 2) / var)
    # Propose a random walk jump for the latent score
    # new_val = old_val + np.random.normal(0.0, leaf_prior_std)
    # new_val = np.random.normal(0.0, leaf_prior_std)
    # V[tree_idx, leaf_to_change] = new_val
    
    return True, leaf_to_change, old_val, penalty



# ==========================================
# 3. LIKELIHOOD AND PRIORS
# ==========================================
@njit(fastmath=True)
def get_tree_prediction(X, F_tree, T_tree, L_tree, R_tree, V_tree):
    n_rows = X.shape[0]
    preds = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        node = 0
        while F_tree[node] != -1:
            if X[i, F_tree[node]] <= T_tree[node]:
                node = L_tree[node]
            else:
                node = R_tree[node]
        preds[i] = V_tree[node]
    return preds

@njit(fastmath=True)
def calc_probit_loglik(Y, partial_pred, tree_pred):
    n_rows = Y.shape[0]
    log_lik = 0.0
    sqrt2 = 1.4142135623730951 
    for i in range(n_rows):
        eta = partial_pred[i] + tree_pred[i]
        p = 0.5 * (1.0 + math.erf(eta / sqrt2))
        if p < 1e-12: p = 1e-12
        elif p > 1.0 - 1e-12: p = 1.0 - 1e-12
        if Y[i] == 1: log_lik += math.log(p)
        else: log_lik += math.log(1.0 - p)
    return log_lik

@njit(fastmath=True)
def calc_grow_prior_ratio(node_depth, leaf_priors):
    # SAFETY 1: If the tree tries to grow deeper than the prior allows, crush it.
    if node_depth + 1 >= len(leaf_priors):
        return -1e9  # Massive negative log-penalty (guarantees rejection)
        
    p_leaf_d = leaf_priors[node_depth]
    p_leaf_d_plus_1 = leaf_priors[node_depth + 1]
    
    p_split = 1.0 - p_leaf_d
    
    # SAFETY 2: Prevent Division by Zero or Log(0)
    if p_leaf_d <= 0.0 or p_split <= 0.0:
        return -1e9
        
    prior_ratio = (p_split * (p_leaf_d_plus_1 ** 2)) / p_leaf_d
    return math.log(prior_ratio)


@njit(fastmath=True)
def mcmc_fractional_sweep(X, Y, F, T, L, R, V, D, node_counts, 
                          sum_trees, alpha_vec, 
                          leaf_priors, leaf_prior_std,
                          step_size,
                          accept_stats,
                          update_fraction=0.1):
    m_trees = F.shape[0]


    # 1. Determine how many trees to mutate
    n_to_update = max(1, int(m_trees * update_fraction))
    
    # 2. Creat a shuffled array of tree indices
    indices = np.arange(m_trees)
    np.random.shuffle(indices)
    trees_to_update = indices[:n_to_update]
    
    # 3. Only loop through the chosen subset
    for m in trees_to_update:
        accept_stats[0] += 1
        old_tree_pred = get_tree_prediction(X, F[m], T[m], L[m], R[m], V[m])
        partial_pred = sum_trees - old_tree_pred
        old_ll = calc_probit_loglik(Y, partial_pred, old_tree_pred)
        
        step_type = np.random.random()
        mutation_type = -1  
        success = False
        penalty = 0.0
        
        if step_type < 0.25:
            success, mutated_node = propose_grow(m, X, F, T, L, R, V, D, node_counts, alpha_vec, leaf_prior_std)
            if success: penalty = calc_grow_prior_ratio(D[m, mutated_node], leaf_priors)
            mutation_type = 0
        elif step_type < 0.50:
            success, mutated_node, old_feat = propose_prune(m, F, L, R, V, node_counts, leaf_prior_std)
            if success: penalty = -calc_grow_prior_ratio(D[m, mutated_node], leaf_priors)
            mutation_type = 1
        elif step_type < 0.75:
            success, mutated_node, old_feat, old_thresh = propose_change(m, X, F, T, node_counts, alpha_vec)
            mutation_type = 2
        else:
            success, mutated_node, old_v, leaf_penalty = propose_leaf_change(m, F, V, node_counts, leaf_prior_std, step_size)
            if success: 
                penalty = leaf_penalty
            mutation_type = 3
            # success, mutated_node, old_v = propose_leaf_change(m, F, V, node_counts, leaf_prior_std)
            # mutation_type = 3
            
        if not success: continue
            
        new_tree_pred = get_tree_prediction(X, F[m], T[m], L[m], R[m], V[m])
        new_ll = calc_probit_loglik(Y, partial_pred, new_tree_pred)
        
        log_alpha = (new_ll - old_ll) + penalty
        accept = math.log(np.random.random()) < log_alpha
        
        if accept:
            sum_trees[:] = partial_pred + new_tree_pred
            accept_stats[1] += 1                        # tracking
        else:
            if mutation_type == 0:
                F[m, mutated_node] = -1
                node_counts[m] -= 2
            elif mutation_type == 1:
                F[m, mutated_node] = old_feat
            elif mutation_type == 2:
                F[m, mutated_node] = old_feat
                T[m, mutated_node] = old_thresh
            elif mutation_type == 3:
                V[m, mutated_node] = old_v
            sum_trees[:] = partial_pred + old_tree_pred

# ==========================================
# 5. PYTHON API HANDOFF
# ==========================================
class FastTreeState:
    def __init__(self, m_trees, init_mean, max_nodes=255):
        self.m_trees = m_trees
        self.max_nodes = max_nodes
        self.F = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.L = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.R = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        
        # THE FIX: Root node starts at Depth 1 to match PyMC's prior array
        self.D = np.ones((m_trees, max_nodes), dtype=np.int8) 
        
        self.T = np.zeros((m_trees, max_nodes), dtype=np.float32)
        self.V = np.zeros((m_trees, max_nodes), dtype=np.float32)
        self.node_counts = np.ones(m_trees, dtype=np.int16)
        self.V[:, 0] = init_mean / m_trees


class FastSurvPGBART(PGBART):
    def __init__(self, vars, target_x=None, 
                 target_y=None, use_sparsity=True, 
                 update_fraction=1.0, 
                 k=2,
                 step_size=.1,
                 return_estimates=False, *args, **kwargs):
        # 1. Initialize the parent class to set up the PyMC hooks
        super().__init__(vars, *args, **kwargs)
        
        self.accept_count = 0
        self.total_count = 0
        self.use_sparsity = use_sparsity
        self.update_fraction = update_fraction
        # self.return_estimates = return_estimates
        
        # 2. THE EXPLICIT OVERRIDE: 
        # Use the injected massive data, or fall back to the graph's self.X / self.Y
        # actual_X = target_x if target_x is not None else self.X
        # actual_Y = target_y if target_y is not None else self.Y
        if target_x is None or target_y is None or return_estimates == True:
            actual_X = self.X
            actual_Y = self.Y
            self.return_estimates = True
            print("returning estimates")
        else:
            self.return_estimates = False
            actual_X = target_x
            actual_Y=target_y
        
        # 3. Setup the C-Engine memory
        base_prob = np.clip(actual_Y.mean(), 1e-5, 1.0 - 1e-5)
        latent_init = sp.stats.norm.ppf(base_prob)
        self.c_tree_state = FastTreeState(m_trees=self.m, init_mean=latent_init)
        
        # self.k = 1.0 # Or whatever tuning you want (lower = less shrinkage)
        # self.leaf_prior_std = np.float32(3.0 / (self.k * np.sqrt(self.m)))
        self.k = k
        self.leaf_prior_std = float(3.0 / (self.k * np.sqrt(self.m)))

        self.step_size = step_size

        self.numba_X = np.copy(np.ascontiguousarray(actual_X))
        self.numba_Y = np.copy(np.ascontiguousarray(actual_Y))
        
        self.flat_sum_trees = np.full(actual_X.shape[0], latent_init, dtype=np.float32)
        
        self.alpha_vec = np.ones(actual_X.shape[1], dtype=np.float32)
        self.numba_leaf_priors = np.array(self.prior_prob_leaf_node, dtype=np.float32)
        self.stats_dtypes = [{
            "variable_inclusion": np.ndarray,
            "tune": bool,
            "F_trees": np.ndarray,
            "T_trees": np.ndarray,
            "L_trees": np.ndarray,
            "R_trees": np.ndarray,
            "V_trees": np.ndarray,
            "node_counts": np.ndarray
        }]
        # tracking only
        self.accept_stats = np.zeros(2, dtype=np.int64)

    def astep(self, q0):
        mcmc_fractional_sweep(
            self.numba_X, self.numba_Y, 
            self.c_tree_state.F, self.c_tree_state.T, 
            self.c_tree_state.L, self.c_tree_state.R, 
            self.c_tree_state.V, self.c_tree_state.D, 
            self.c_tree_state.node_counts,
            self.flat_sum_trees, 
            self.alpha_vec, 
            self.numba_leaf_priors,
            self.leaf_prior_std,
            self.step_size,
            self.accept_stats,
            self.update_fraction # <-- Pass to C-Loop
        )
        
        var_counts = get_split_counts(self.c_tree_state.F, self.num_variates)
        
        if self.tune and self.use_sparsity:
            if self.iter > self.m:
                self.alpha_vec += var_counts.astype(np.float32)
        
        stats = {
            "variable_inclusion": var_counts, 
            "tune": self.tune,
            "F_trees": self.c_tree_state.F.copy(),
            "T_trees": self.c_tree_state.T.copy(),
            "L_trees": self.c_tree_state.L.copy(),
            "R_trees": self.c_tree_state.R.copy(),
            "V_trees": self.c_tree_state.V.copy(),
            "node_counts": self.c_tree_state.node_counts.copy()
        }
       
        if self.return_estimates == True:
            self.sum_trees[0, 0, :] = self.flat_sum_trees
            return self.sum_trees, [stats]
        else:
            return np.zeros(1), [stats]
            


@njit(fastmath=True)
def _fast_predict_trace(X, F, T, L, R, V):
    n_samples = F.shape[0]
    m_trees = F.shape[1]
    n_obs = X.shape[0]
    preds = np.zeros((n_samples, n_obs), dtype=np.float32)
    for s in range(n_samples):
        for i in range(n_obs):
            tree_sum = 0.0
            
            # Drop this observation down all M trees in this specific sample
            for m in range(m_trees):
                node = 0
                
                # Walk the tree until we hit a leaf (F == -1)
                while F[s, m, node] != -1:
                    feat = F[s, m, node]
                    val = X[i, feat]
                    
                    if val <= T[s, m, node]:
                        node = L[s, m, node]
                    else:
                        node = R[s, m, node]
                        
                tree_sum += V[s, m, node]
                
            preds[s, i] = tree_sum            
    return preds


@njit(fastmath=False, parallel=False)
def _fast_predict_trace(X, F, T, L, R, V):
    n_samples = F.shape[0]
    m_trees = F.shape[1]
    n_obs = X.shape[0]
    
    # Pre-allocate output matrix
    preds = np.zeros((n_samples, n_obs), dtype=np.float32)
    
    # LOOP INVERSION: Put observations inside the sample/tree loops!
    for s in range(n_samples):
        for m in range(m_trees):
            # The tree structure for [s, m] is now locked into the CPU cache line!
            f_tree = F[s, m]
            t_tree = T[s, m]
            l_tree = L[s, m]
            r_tree = R[s, m]
            v_tree = V[s, m]
            
            # Vector-friendly sweep over all observations
            for i in range(n_obs):
                node = 0
                while f_tree[node] != -1:
                    feat = f_tree[node]
                    if X[i, feat] <= t_tree[node]:
                        node = l_tree[node]
                    else:
                        node = r_tree[node]
                        
                preds[s, i] += v_tree[node]
                
    return preds

def predict_out_of_sample(trace, X_new, m_trees=50, max_nodes=255, thin=1):
    """
    Extracts the serialized forest topology from a PyMC trace, 
    applies an optional thinning parameter to accelerate prediction,
    and generates predictions.
    
    Parameters:
    -----------
    thin : int
        Take every `thin` sample from the posterior. 
        e.g., thin=10 reduces 4,000 samples to 400 for a 10x speedup.
    """
    print("Starting predict layout...")
    stats = trace.sample_stats
    
    # Calculate initial raw n_samples using values.size directly
    n_samples_raw = stats["F_trees"].values.size // (m_trees * max_nodes)
    
    # ZERO-ALLOCATION FIX + SLICING VIA MEMORY VIEWS
    # By using [::thin], we slice the reshaped tensor before copy=False checks types.
    # NumPy passes this sliced stride straight down to Numba as a contiguous view.
    F = np.ascontiguousarray(stats["F_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    T = np.ascontiguousarray(stats["T_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.float32, copy=False)
    L = np.ascontiguousarray(stats["L_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    R = np.ascontiguousarray(stats["R_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    V = np.ascontiguousarray(stats["V_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.float32, copy=False)
    
    # Deduce the thinned sample count that Numba will actually process
    n_samples_thinned = F.shape[0]
    
    X_numba = np.ascontiguousarray(X_new, dtype=np.float64)
    n_obs = X_new.shape[0]
    
    print(f"Processing: Dropping {n_obs} rows down {n_samples_thinned} thinned historical forests (Thin factor: {thin})...")
    latent_scores = _fast_predict_trace(X_numba, F, T, L, R, V)
    
    # Vectorized fast probit translation
    probabilities = sp.stats.norm.cdf(latent_scores)
    
    return probabilities