#%%

# %%
from asyncio import ALL_COMPLETED
from collections import Counter

import numpy as np
#%%
from scipy.stats import expon

#%%

N = 10
nodes = np.arange(N)


# %%
def simulate_rand(n_events=10, N=10):

    t = 0
    h = []
    for _ in range(n_events):
        t += np.random.exponential()
        h.append([np.random.randint(N), np.random.randint(N), t])


#%%
socialities = np.exp(np.arange(10))

socialities /= np.sum(socialities)

socialities


#%%
def intensity(i, j, t):
    return np.exp(np.cos((socialities[i] + socialities[j]) * t))


#%%
def simulate_thinning(intensity, N, count_max=10000):

    all_edges = np.array([[i, j] for i in range(N) for j in range(i + 1, N)])
    linear_indices = np.arange(len(all_edges))

    def all_intensities(intensity_func, t):
        return np.array([intensity(i, j, t) for (i, j) in all_edges])

    def total_intensity(t):
        return sum(all_intensities(intensity, t))

    t = 0
    h_simul = []
    count = 0
    while (True and count <= count_max):
        count += 1
        lambda_star = total_intensity(t)
        q = expon(lambda_star).rvs(1)
        candidate = t + q

        acceptance_rate = total_intensity(candidate) / lambda_star
        if (np.random.uniform() <= acceptance_rate):
            ps = all_intensities(intensity, t).ravel()
            ps /= np.sum(ps)
            assert ps.shape[0] == linear_indices.shape[0]
            edge = all_edges[np.random.choice(linear_indices, p=ps)]
            t = candidate
            h_simul.append([edge[0], edge[1], t])
        else:
            continue

    return h_simul


#%%

if __name__ == '__main__':
    h_simul = simulate_thinning(intensity, 10)
    import ipdb
    ipdb.set_trace()
# %%
