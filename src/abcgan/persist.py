"""
This module supports persistence of the generator and discriminator.

It saves two files a parameters file and a configuration file.

It also supports persisting of multiple modules.

To be persistable in this way the module must have a property
containing a json serializable input dictionary as
mdl.input_args
"""
from abcgan import bv_model
from abcgan import hfp_model
from abcgan import mean_estimation as me
import json
import torch
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'models')


def fullname(inst):
    tp = type(inst)
    # check if instance is actually a type
    if isinstance(inst, type):
        tp = inst
    return tp.__module__ + '.' + tp.__name__


# list types of persistable object
types = [
    bv_model.Generator,
    bv_model.Critic,
    bv_model.Driver_Generator,
    bv_model.Driver_Critic,
    me.Transformer,
    hfp_model.HFP_Critic,
    hfp_model.HFP_Generator,
    hfp_model.HFP_Transformer
]
# register types of objects that can be persisted
type_dict = {fullname(t): t for t in types}


def persist(generator, critic, name='gan', dir_path=dir_path):
    """
    Persists abcgan generator and critic modules.

    Persists both input arguments and parameters.

    Parameters:
        generator: torch.nn.Module
            module for the generator
        critic: torch.nn.Module
            module for the critic
        name: str, optional
            name of the saved configuration
        dir_path: str, optional
            default is the models directory. None assumes
            file is in local directory.

    The generator, critic and any transformers passed in as
    arguments to these must be registered in persist.py and must
    have a parameter 'input_args' that specifies their input
    arguments as a dictionary
    """
    if dir_path is not None:
        name = os.path.join(dir_path, name)
    jdict = {}
    if hasattr(generator, 'transformer'):
        jdict['gen_trans_input'] = generator.transformer.input_args
        jdict['gen_trans_type'] = fullname(generator.transformer)
    if hasattr(critic, 'transformer'):
        jdict['crit_trans_input'] = critic.transformer.input_args
        jdict['crit_trans_type'] = fullname(critic.transformer)
    jdict['gen_input'] = generator.input_args
    jdict['gen_type'] = fullname(generator)
    jdict['crit_input'] = critic.input_args
    jdict['crit_type'] = fullname(critic)
    # verify types are registered
    for k, v in jdict.items():
        if k.endswith('type'):
            if v not in type_dict:
                raise ValueError(f"Uknown module type {v}.")
    with open(name+'.json', 'w+') as f:
        json.dump(jdict, f, indent=2)
    state_dict = {}
    state_dict['gen'] = generator.state_dict()
    state_dict['crit'] = critic.state_dict()
    torch.save(state_dict, name + '.pt')


def recreate(name='gan', dir_path=dir_path):
    """
    Load a pre-trained generator and discriminator.

    Parameters
    --------------
    name: str, optional
        name of the configuration to load, as saved by
        persist. default: 'wgan_gp'
    dir_path: str, optional
        default is the models directory. None assumes
        file is in local directory.

    Returns
    -------------
    generator: torch.nn.module
        the loaded generator
    critic: torch.nn.module
        the loaded critic

    Modules must have previosuly been saved. All modules are
    loaded on the cpu, they can subsequently be moved.
    """
    if dir_path is not None:
        name = os.path.join(dir_path, name)
    with open(name + '.json', 'r') as f:
        jdict = json.load(f)
    if 'gen_trans_type' in jdict:
        trans_in = jdict['gen_trans_input']
        trans_type = type_dict[jdict['gen_trans_type']]
        jdict['gen_input']['transformer'] = trans_type(**trans_in)
    if 'crit_trans_type' in jdict:
        trans_in = jdict['crit_trans_input']
        trans_type = type_dict[jdict['crit_trans_type']]
        jdict['crit_input']['transformer'] = trans_type(**trans_in)
    gen_in = jdict['gen_input']
    gen_type = type_dict[jdict['gen_type']]
    generator = gen_type(**gen_in)
    crit_in = jdict['crit_input']
    crit_type = type_dict[jdict['crit_type']]
    critic = crit_type(**crit_in)
    state_dict = torch.load(name + '.pt',
                            map_location=torch.device('cpu'))
    generator.load_state_dict(state_dict['gen'])
    critic.load_state_dict(state_dict['crit'])

    return generator, critic
