import torch

def generate_fixed_entry_data(fixed_mask, initial_values):
    """
    Preprocess to construct data to efficiently apply fixed values (e.g. pinned vertices, boundary conditions) to a vector.

    Args:
        fixed_mask (torch.BoolTensor): Length-N mask where True = fixed.
        initial_values (torch.Tensor): Length-N tensor of values; entries for non-fixed indices are ignored.

    Returns:
        fixed_inds (torch.LongTensor)
        unfixed_inds (torch.LongTensor)
        fixed_values (torch.Tensor)
        unfixed_values (torch.Tensor)
    """
    fixed_inds = torch.nonzero(fixed_mask, as_tuple=False).squeeze(1)
    unfixed_inds = torch.nonzero(~fixed_mask, as_tuple=False).squeeze(1)

    fixed_values = initial_values[fixed_mask]
    unfixed_values = initial_values[~fixed_mask]

    return fixed_inds, unfixed_inds, fixed_values, unfixed_values


def apply_fixed_entries(fixed_inds, unfixed_inds, fixed_values, unfixed_values):
    """
    Applies fixed values to a vector, using indexing arrays from generate_fixed_entry_data().

    Args:
        fixed_inds: LongTensor of fixed indices
        unfixed_inds: LongTensor of unfixed indices
        fixed_values: Tensor or scalar for fixed entries
        unfixed_values: Tensor or scalar for unfixed entries
    """
    dtype = torch.float32
    if not torch.is_tensor(fixed_values):
        fixed_values = torch.tensor(fixed_values, dtype=dtype)
    else:
        fixed_values = fixed_values.to(dtype=dtype)

    if not torch.is_tensor(unfixed_values):
        unfixed_values = torch.tensor(unfixed_values, dtype=dtype)
    else:
        unfixed_values = unfixed_values.to(dtype=dtype)

    N = fixed_inds.numel() + unfixed_inds.numel()
    out = torch.zeros(N, dtype=torch.float32, device=fixed_values.device)

    out[fixed_inds] = fixed_values
    out[unfixed_inds] = unfixed_values

    return out
