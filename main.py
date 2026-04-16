import yaml
import os
import sys
import utils as U
import wandb
from models.model_runners import *
from accelerate import Accelerator

accelerator = Accelerator()


def replace_previous_settings(prev_config, update_config):
    
    # Updates prev_config with new_config and returns prev_config
    # Replace previous settings with new ones
    prev_config = prev_config.copy()
    update_config = update_config.copy()
    for key in [
        'epochs', 'prev_wts', 'load_last', 'lr_schedule',
        'lr_warmup_start', 'lr_warmup_end', 'lr_warmup_steps',
        'lr_flat_steps', 'lr_floor', 'lr_decay_steps','eval_frequency',
        'loader', 'log_wandb', 'eval_only', 'batch_size', 'rl', 'save_weights',
        'top_peaks', 'classifier_config', 'new_exp', 'inference',
    ]:
        if key == 'loader':
            # These must be consistent with embedding layer in decoder
            update_config[key]['synonyms'] = prev_config[key]['synonyms']
            update_config[key]['dictionary_path'] = prev_config[key]['dictionary_path']
            update_config[key]['reverse'] = prev_config[key]['reverse']
        prev_config[key] = update_config[key]
    prev_config['decoder_mdlm']['cbg'] = update_config['decoder_mdlm']['cbg']
    
    return prev_config

def replace_previous_for_eval_only(config, evconfig):
    config = config.copy()
    config['loader']['train_dataset_path'] = evconfig['eval_only']['eval_dataset_path']
    config['loader']['train_name'] = None
    config['loader']['val_dataset_path'] = evconfig['eval_only']['eval_dataset_path']
    config['loader']['datapath_extension'] = evconfig['eval_only']['datapath_extension']
    config['loader']['val_name'] = evconfig['eval_only']['eval_name']
    cc = evconfig['eval_only']['loader_custom_columns']
    config['loader']['custom_columns'] = [] if cc == None else cc
    config['loader']['val_steps'] = evconfig['eval_only']['val_steps']
    config['loader']['disperse'] = evconfig['eval_only']['disperse']
    return config

def main():
     ##############
    # Read yamls #
    ##############

    #######################
    # Configuration files #
    #######################

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./yaml/config.yaml"

    #######################
    # Configuration files #
    #######################

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./yaml/config.yaml"

    # Read yamls
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    # Overrides over a loaded previous experiment
    config_ = config.copy()
    # Eval config will not be overwritten
    with open("./yaml/eval.yaml") as stream:
        evconfig = yaml.safe_load(stream)

    ########################################################
    # Create experiment directory in save/downstream_only/ #
    ########################################################

    # Continuing previous downstream run; maybe create new directory
    timestamp = U.timestamp()
    if config['prev_wts'] is not None:
        # Set the read directory to properly load previous model weights
        rddir = os.path.join(config['prev_wts'])
        # Reconcile the loaded model's config with the current config in yaml/config.yaml
        with open(os.path.join(config['prev_wts'], "yaml", "config.yaml")) as stream:
            config = yaml.safe_load(stream)
        config = replace_previous_settings(config, config_)
        # Set the save directory. Perhaps create a new project directory?
        if not config['save_weights']:
            svdir = './'
        elif config['new_exp']:
            svdir = os.path.join('save', timestamp)
            if (not config['eval_only']) and accelerator.is_local_main_process:
                U.create_experiment(svdir, svwts=config['save_weights'], config=config)
                print("<MAINCOMMENT> Experiment is writing to directory %s"%svdir)
        else:
            svdir = os.path.join(config['prev_wts'])
            timestamp = config['prev_wts']    
        
    # Create new experiment without previous weights
    elif config['save_weights'] and (not config['eval_only']) and accelerator.is_local_main_process:
        rddir = None
        svdir = os.path.join('save', timestamp)
        if accelerator.is_main_process:
            U.create_experiment(svdir, svwts=config['save_weights'])
            print("<MAINCOMMENT> Experiment is writing to directory %s"%svdir)
    else:
        rddir = None
        svdir = './'

    # Eval only. Must set before loader is created.
    if config['eval_only']:
        config = replace_previous_for_eval_only(config, evconfig)
    
    #####################
    # Downstream object #
    #####################

    print("<MAINCOMMENT> Denovo sequencing")
    if 'diff' in config['decoder_name']:
        print("<MAINCOMMENT> Using diffusion decoder")
        D = DenovoDiffusionObj(config, svdir=svdir, rddir=rddir)
    elif 'mdlm' in config['decoder_name']:
        print("<MAINCOMMENT> Using masked diffusion language decoder")
        D = DenovoMDLMObj(config, svdir=svdir, rddir=rddir)
    elif 'd3pm' in config['decoder_name']:
        print("<MAINCOMMENT> Using D3PM")
        D = DenovoD3PMObj(config, svdir=svdir, rddir=rddir)
    else:
        print("<MAINCOMMENT> Using autoregressive decoder")
        D = DenovoArObj(config, svdir=svdir, rddir=rddir)

    # WandB
    if config['log_wandb'] and (config['eval_only'] == False) and accelerator.is_local_main_process:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
			config={
				'master': config,
                'save_directory': timestamp,
                'model_parameters': D.model.total_params(),
			},
		)   
    
    ##################################
    # Run training and/or evaluation #
    ##################################

    if config['eval_only']:
        evc = evconfig['eval_only']
        
        # Apply settings that are independent of training
        max_batches = int(eval(str(evc['val_steps'] if evc['val_steps'] is not None else 9e10)))
        if 'max_batches' in evc.keys(): max_batches = evc['max_batches'] # override val steps
        if config['decoder_name'] in ['diff', 'mdlm']:
            if evc['clamp_denoised'] is not None:
                D.model.decoder.clamp_denoised = evc['clamp_denoised']
            if evc['n'] is not None:
                D.model.ens_size = evc['n']
                D.eval_kwargs['n'] = D.model.ens_size
        
        # Turn gradients off for de novo model
        for parm in D.model.parameters(): parm.requires_grad=False
        
        # Classifier guidance
        no_grad = False if hasattr(D, 'classifier') else True

        # Run evaluation
        evalkwargs = dict(evc['eval_kwargs']) if evc['eval_kwargs'] is not None else {}
        if config['inference']:
            D.inference(
                output_filename=evc['outpath'],
                dset=evc['set'],
                max_batches=max_batches,
                stream_write=evc['stream'],
                no_grad=no_grad,
                kwargs=D.eval_kwargs|evalkwargs,
                save_keys=evconfig['inference_save_keys'],
            )
        else:
            out, df = D.evaluation(
                dset=evc['set'], 
                max_batches=max_batches, 
                save_df=evc['save'], 
                stream_write=evc['stream'],
                no_grad=no_grad, 
                kwargs=D.eval_kwargs|evalkwargs,
            )
        
            # Saving results
            if evc['save']:
                eval_out_path = evc['outpath'] if evc['outpath'] is not None else os.path.join(svdir, "output.parquet")
                if evc['stream']:
                    os.system(f"mv ./hold.parquet {eval_out_path}")
                else:
                    df.to_parquet(eval_out_path)
            print("\n", out)
    else:
        if accelerator.is_local_main_process:
            print("Test validation", end='')
            out, _ = D.evaluation(dset='val', max_batches=2, kwargs=D.eval_kwargs)
            assert D.config['high_score'] in out.keys()
            print("\rTest validation passed")
        accelerator.wait_for_everyone()
        print(D.TrainEval()[-1])

if __name__ == '__main__':
    main()
