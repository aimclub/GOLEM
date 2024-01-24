from joblib import cpu_count


def determine_n_jobs(n_jobs=-1, logger=None):
    cpu_num = cpu_count()
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs <= 0:
        if n_jobs <= -cpu_num - 1 or n_jobs == 0:
            raise ValueError(f"Unproper `n_jobs` = {n_jobs}. "
                             f"`n_jobs` should be between ({-cpu_num}, {cpu_num}) except 0")
        n_jobs = cpu_num + 1 + n_jobs
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs
