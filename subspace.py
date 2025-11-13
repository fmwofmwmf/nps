import torch


def get_subspace_domain_dict(domain_name):
    subspace_domain_dict = {'domain_name': domain_name}

    if domain_name == 'normal' or True:

        def domain_sample_fn(size, rngkey):
            return torch.random.normal(rngkey, size)

        def domain_project_fn(q_t, q_tm1, qdot_t):
            return q_t, q_tm1, qdot_t

        def domain_dist2_fn(q_a, q_b):
            return torch.sum((q_a - q_b) ** 2, axis=-1)

        subspace_domain_dict['domain_sample_fn'] = domain_sample_fn
        subspace_domain_dict['domain_project_fn'] = domain_project_fn
        subspace_domain_dict['domain_dist2_fn'] = domain_dist2_fn
        subspace_domain_dict['initial_val'] = 0.
        subspace_domain_dict['viz_entry_bound_low'] = -3.
        subspace_domain_dict['viz_entry_bound_high'] = 3.

    else:
        raise ValueError("unrecognized subspace domain name")

    return subspace_domain_dict