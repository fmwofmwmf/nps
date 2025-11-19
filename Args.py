class Args:
    model_type = "SubspaceMLP"
    activation = "ELU"
    MLP_hidden_layers = 5
    MLP_hidden_layer_width = 128
    subspace_dim = 12

    n_train_iters = 500000
    batch_size = 128
    lr = 1e-4
    lr_decay_every = 250000
    lr_decay_frac = 0.5

    weight_expand = 0.5
    sigma_scale = 1e-3
    expand_type = "iso"

    report_every = 100000
    output_dir = "./checkpoints"

    system_name = "FEM"
    problem_name = "bistable"
    subspace_domain_type = "normal"

    subspace_model = "./checkpoints/SubspaceMLP__final_nodrop"
    integrator = "implicit-proximal"