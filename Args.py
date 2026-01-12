class Args:
    model_type = "SubspaceMLP"
    activation = "ELU"
    MLP_hidden_layers = 5
    MLP_hidden_layer_width = 128
    subspace_dim = 2
    shape_space_dim = 3

    n_train_iters = 250000
    batch_size = 32
    lr = 1e-4
    lr_decay_every = 50000
    lr_decay_frac = 0.5

    weight_expand = 1
    sigma_scale = 1
    expand_type = "iso"

    report_every = 50000
    output_dir = "./checkpoints"

    system_name = "FEM"
    problem_name = "bistable"
    subspace_domain_type = "normal"

    subspace_model = "./checkpoints/SubspaceMLP_200000"
    integrator = "implicit-proximal"