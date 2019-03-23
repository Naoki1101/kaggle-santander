from imblearn.under_sampling import RandomUnderSampler


def downsampling(x_train, y_train, seed):
    sampler = RandomUnderSampler(random_state=seed, replacement=True)
    X_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    return X_resampled, y_resampled
