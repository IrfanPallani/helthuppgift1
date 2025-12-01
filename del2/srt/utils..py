import numpy as np
import scipy.stats as st


def normal_ci(data, alpha=0.05):
    data = np.asarray(data)
    data = data[-np.isnan(data)]
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    z = st.norm.ppf(1 - alpha/2)
    return mean, mean - z*se, mean + z*se


def bootstrap_ci(data, n_boot=2000, alpha=0.05, statfunc=np.mean, seed=42):

    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    data = data[-np.isnan(data)]

    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boots.append(statfunc(sample))
    lower = np.percentile(boots, 100*(alpha/2))
    upper = np.percentile(boots, 100*(1-alpha/2))
    return statfunc(data), lower, upper


def proportion_confint(count, nobs, alpha=0.05):
    if nobs == 0:
        return np.nan, np.nan, np.nan
    prop = count / nobs
    z = st.norm.ppf(1 - alpha/2)
    se = np.sqrt(prop * (1-prop) / nobs)
    return prop, max(0, prop - z*se), min(1, prop + z*se)
