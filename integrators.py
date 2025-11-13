import torch

def state_style(method):
    if method in ['implicit-proximal']:
        return '2-state'
    elif method in []:
        return 'state-vel'
    else:
        raise ValueError("unrecognized method")

def initialize_integrator(int_opts, int_state, new_method):

    def soft_add(key, value):
        if key not in int_opts:
            int_opts[key] = value

    int_opts['method'] = new_method

    soft_add('timestep_h', 0.05)

    # TODO do something about velocity / 2-state style swtiching?

    if int_opts['method'] == 'implicit-proximal':
        soft_add('solver_type', 'JAX-LBFGS')

    else:
        raise ValueError("unrecognized integrator method")


def update_state(int_opts, int_state, new_q, with_velocity):

    this_style = state_style(int_opts['method'])

    if this_style == '2-state':
        int_state['q_t'] = new_q

        if with_velocity:
            # leave int_state['q_tm1'] untouched
            pass
        else:
            int_state['q_tm1'] = new_q

    elif this_style == 'state-vel':
        if with_velocity:
            int_state['qdot_t'] = (new_q - int_state['q_t']) / int_opts['timestep_h']
        else:
            int_state['qdot_t'] = torch.zeros_like(int_state['qdot_t'])

        int_state['q_t'] = new_q

    else:
        raise ValueError("unrecognized style")


def apply_domain_projection(int_state, subspace_domain_dict):
    if subspace_domain_dict is None: return

    int_state['q_t'], int_state['q_tm1'], int_state['qdot_t'] = subspace_domain_dict[
        'domain_project_fn'](int_state['q_t'], int_state['q_tm1'], int_state['qdot_t'])