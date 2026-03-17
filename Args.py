class Args:
    config_file = "configs/bar_thingy.json"
    # exp, imp, imp_torch, imp_semi
    integrator_name = "imp"
    use_subspace = True
    record_frames = 100
    ground_truth_offset = (-3, 0, 0)

    experiment_id = "p2"
    subspace_model = "180000"