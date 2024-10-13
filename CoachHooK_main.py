from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from EasyEdit.easyeditor.models.rome.layer_stats import layer_stats
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.util.generate import generate_fast
from EasyEdit.easyeditor.util.globals import *

from EasyEdit.easyeditor.models.memit.compute_ks import compute_ks
from EasyEdit.easyeditor.models.memit.compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from EasyEdit.easyeditor.models.memit.memit_hparams import MEMITHyperParams
from EasyEdit.easyeditor.models.memit.memit_main import get_cov, upd_matrix_match_shape, get_context_templates

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


def apply_CoachHooK_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    alpha_z: int,
    eval_interval: Optional[int] = None,
    cache_template: Optional[str] = None,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
        
    def recording_hook(new_module:torch.nn.Module, name:str):
        def hook(module, inputs, outputs):
            # print(inputs, outputs)
            x = inputs[0]
            # print(x.shape)
            new_module.to(device=x.device, dtype=x.dtype)
            new_outputs = new_module(x)
            
            diff = new_outputs - outputs
            norm_diff = diff.norm(dim=-1)
            
            z_scores_diff = (norm_diff - norm_diff.mean(dim=-1,keepdim=True)) / norm_diff.std(dim=-1,keepdim=True)
            
            statis[name].append(z_scores_diff.detach().cpu())
        return hook
    
    statis, hook_handles, accu_params = {}, {}, {}
    accu_params_set = []
    for i,chunk in enumerate(_chunks(requests, hparams.batch_size)):
        accu_params = execute_CoachHooK(model, tok, chunk, hparams, alpha_z, accu_params, cache_template)
        # print(accu_params)

        for m_name, params in accu_params.items():
            statis[m_name] = []
            # statis[m_name] = {"max_scores_diff":[], "max_scores_accu":[]}
            module = nethook.get_module(model, m_name)
            new_module = accu_params[m_name]["new_weight"]
            # C_ks_accu = accu_params[m_name]["C_ks_accu"]
            hook_handle = module.register_forward_hook(recording_hook(new_module, m_name))
            hook_handles[m_name] = hook_handle  
        
        prompts = []
        for request in chunk:
            prompts.append(request["prompt"])
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        model.eval()
        with torch.no_grad():
            model(**inputs)            
        for hook_handle in hook_handles.values():
            hook_handle.remove()
            
        for m_name, z_scores in statis.items():
            max_scores = z_scores[0].max(dim=-1).values.to(torch.float)
            accu_params[m_name]["alpha"] = min(max_scores.quantile(dim=-1,q=0).item(), accu_params[m_name]["alpha"])
        
        if eval_interval is not None:
            edit_num = (i+1)*hparams.batch_size
            if edit_num % eval_interval == 0:
                accu_params_set.append(deepcopy(accu_params))

    
    if (eval_interval is not None and edit_num % eval_interval != 0) or eval_interval is None:
        accu_params_set.append(accu_params)
    
    print(f"New weights successfully inserted into {list(accu_params.keys())}")

    return accu_params_set



def execute_CoachHooK(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    alpha_z: int,
    accu_params: Dict, #Accumulated keys and weight changes
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the comeba-hk update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
 
    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
 
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                    print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
 
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
 
    for request in requests[:10]:
        print(
            f"comeba-hk request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )
 
    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
 
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
    
    
    for i, layer in enumerate(hparams.layers):
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        if module_name not in accu_params.keys():
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
                hparams=hparams
            )
            
            m = deepcopy(nethook.get_module(model, module_name))
            m.weight = torch.nn.parameter.Parameter(weights_copy[weight_name])
            accu_params[module_name] = {
                                        "C_accu": hparams.mom2_update_weight * cov,
                                        "new_weight": m,
                                        "alpha": alpha_z,
                                        }

    
    def validated_hook(new_module:torch.nn.Module, alpha:torch.Tensor):
        def hook(module, inputs, outputs):
            # print(inputs, outputs)
            x = inputs[0]
            # print(x.shape)
            new_module.to(device=x.device, dtype=x.dtype)
            new_outputs = new_module(x)
            
            diff = new_outputs - outputs
            norm_diff = diff.norm(dim=-1)
            
            z_scores = (norm_diff - norm_diff.mean(dim=-1,keepdim=True)) / norm_diff.std(dim=-1,keepdim=True)
            
            idx_bool = torch.ge(z_scores, alpha)
            
            outputs[idx_bool,:] = new_outputs[idx_bool,:]
            return outputs
        return hook
    
    def temporary_hook(new_module:torch.nn.Module):
        def hook(module, inputs, outputs):
            # print(inputs, outputs)
            # print(new_weight)
            x = inputs[0]
            new_module.to(device=x.device, dtype=x.dtype)
            new_outputs = new_module(x)
            return new_outputs
        return hook


    hook_handles = {}
    #hang validated hook to calculate the optimized keys.
    for layer in hparams.layers:
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        module = nethook.get_module(model, module_name)
        new_module = accu_params[module_name]["new_weight"]
        alpha = accu_params[module_name]["alpha"]
        hook_handle = module.register_forward_hook(validated_hook(new_module, alpha))
        hook_handles[module_name] = hook_handle
        
    
    
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
                print("Load z cache file successfully")
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
 
        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
 
            z_list.append(cur_z)
 
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")          
    zs = torch.stack(z_list, dim=1)
    
    
    for hook_handle in hook_handles.values():
        hook_handle.remove()
    
    for layer in hparams.layers:
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        module = nethook.get_module(model, module_name)
        new_module = accu_params[module_name]["new_weight"]
        hook_handle = module.register_forward_hook(temporary_hook(new_module))
        hook_handles[module_name] = hook_handle

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        
        # Get current model activations
        # token_idxs = idxs_context_templates
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into hook layer {layer}")

        
        # Compute residual error
        # token_idxs = idxs_requests
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        targets = zs - cur_zs
        # print("zs:", zs)
        # print("cur_zs:", cur_zs)
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        
 
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        
        
        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )
        
        
        C_accu, new_weight = (accu_params[module_name]["C_accu"].double().to(layer_ks.device), 
                              accu_params[module_name]["new_weight"].weight.double().to(layer_ks.device))
        
        # C_ks_accu = accu_params[module_name]["C_ks_accu"].double().to(layer_ks.device)
        
        C_ks = layer_ks @ layer_ks.T
        
        adj_k = torch.linalg.solve(
            C_accu + C_ks,
            layer_ks,
        )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T
        
        
        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        # Accumulation
        C_accu += C_ks
        new_weight += upd_matrix
        # C_ks_accu += C_ks
        
        accu_params[module_name]["C_accu"] = C_accu.detach().float().cpu()
        accu_params[module_name]["new_weight"].weight = torch.nn.parameter.Parameter(new_weight.detach().float())
        accu_params[module_name]["new_weight"].to("cpu")
        
        
        #update the new weight in the hook
        hook_handles[module_name].remove()
        module = nethook.get_module(model, module_name)
        new_module = accu_params[module_name]["new_weight"]
        alpha = accu_params[module_name]["alpha"]
        hook_handles[module_name] = module.register_forward_hook(validated_hook(new_module, alpha))
    
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
                   
            
        # Clear GPU memory
        for x in [layer_ks, cur_zs, targets, upd_matrix, C_accu, C_ks, new_weight]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
        
    for hook_handle in hook_handles.values():
        hook_handle.remove()
        
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return accu_params




def apply_CoachHooK_wohk_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    eval_interval: Optional[int] = None,
    return_orig_weights=True,
    cache_template: Optional[str] = None,
    keep_original_weight=True,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    accu_params = {}
    accu_params_set = []
    for i,chunk in enumerate(_chunks(requests, hparams.batch_size)):
        deltas, accu_params = execute_CoachHooK_wohk(model, tok, chunk, hparams, accu_params, cache_template=cache_template)
    
        with torch.no_grad():
            for w_name, (key_mat, val_mat) in deltas.items():
                key_mat, val_mat = key_mat.to(f"cuda:{hparams.device}"), val_mat.to(f"cuda:{hparams.device}")
                upd_matrix = key_mat @ val_mat.T
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
    
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone().cpu()
                w[...] += upd_matrix.float()
            
            if eval_interval is not None:
                edit_num = (i+1)*hparams.batch_size
                if edit_num % eval_interval == 0:
                    for w_name,w in weights_copy.items():
                        m_w = nethook.get_parameter(model, w_name)
                        accu_params[w_name]["w"] = m_w.cpu().clone()
                    accu_params_set.append(deepcopy(accu_params))

    if (eval_interval is not None and edit_num % eval_interval != 0) or eval_interval is None:
        for w_name,w in weights_copy.items():
            m_w = nethook.get_parameter(model, w_name)
            accu_params[w_name]["w"] = m_w.cpu().clone()
        accu_params_set.append(accu_params)
        
    print(f"New weights successfully inserted into {list(deltas.keys())}")
    
    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy, accu_params_set




def execute_CoachHooK_wohk(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    accu_params: Dict,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    for request in requests[:10]:
        print(
            f"comeba-hk without hook request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )
        
        if weight_name not in accu_params.keys():
            accu_params[weight_name] = {"C_accu": cov}
        C_accu = accu_params[weight_name]["C_accu"].to(layer_ks.device)

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * C_accu.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T
        
        C_accu += layer_ks @ layer_ks.T
        accu_params[weight_name]["C_accu"] = C_accu.cpu()

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, accu_params

