# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:22:12 2023

@author: Li_Sh
"""

import os
import torch
import time
from EasyEdit.easyeditor.util import nethook
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from CoachHooK_main import apply_CoachHooK_to_model, MEMITHyperParams, apply_CoachHooK_wohk_to_model, _chunks
import numpy as np
from EasyEdit.easyeditor.evaluate.evaluate import compute_edit_quality
import argparse



os.environ['TOKENIZERS_PARALLELISM'] = "false"


def main():

    parser = argparse.ArgumentParser(description='CoachHook implementation.')
    parser.add_argument('--stats_dir', type=str, default="./data/stats", help='the path of stat file')
    parser.add_argument('--model', type=str, choices=["EleutherAI/gpt-j-6B","gpt2-xl"], default="gpt2-xl", help='the model to use')
    parser.add_argument("--ds_dir", type=str, choices=["./editing-data/data/zsre/zsre_mend_eval.json", "./editing-data/data/counterfact/counterfact-edit.json"], 
                        default="./editing-data/data/zsre/zsre_mend_eval.json", help="the dataset directory")
    parser.add_argument('--consecutive', type=int, choices=[0,1], default=0, help='whether the editing happens in a consecutive way')
    parser.add_argument('--bsz', type=int, default=30, help='the editing batch size to use')
    parser.add_argument('--mom2_update_weight', type=int, default=15000, help='the moment2 update weight')
    parser.add_argument('--alpha_z', type=float, default=2.2, help='the initial alpha')
    parser.add_argument('--num_layers', type=int, choices=range(1,8), default=8, help='the number of layers in the critical path to use')
    parser.add_argument('--sample', type=int, default=100, help='number of samples to use')
    parser.add_argument('--wohk', type=int, choices=[0,1], default=0, help='whether to use the hook layer')
    args = parser.parse_args()


    model_id = args.model


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    if "gpt-j-6B" in model_id:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision="float16", torch_dtype=torch.float16).to("cuda")
        hparams = MEMITHyperParams.from_hparams('./EasyEdit/hparams/MEMIT/gpt-j-6B.yaml')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        hparams = MEMITHyperParams.from_hparams('./EasyEdit/hparams/MEMIT/gpt2-xl.yaml')  
    print(model)


    hparams.stats_dir = args.stats_dir
    hparams.layers = hparams.layers[-args.num_layers:]
    hparams.batch_size = args.bsz
    consecutive = args.consecutive
    hparams.mom2_update_weight = args.mom2_update_weight
    alpha_z = args.alpha_z
    eval_interval = 20*hparams.batch_size
    if "counterfact-edit" in args.ds_dir:
        ds_name = "counterfact-edit"
    else:
        ds_name = "zsre_mend_eval"
    editing_method = "CoachHooK"
    model_name = model_id.rpartition("/")[-1]
    sample = args.sample
    # print(hparams)
    wohk = args.wohk




    if ds_name == "counterfact-edit":
        with open(args.ds_dir) as f:
            ds = json.load(f)
        requests = ds[:sample]
        for record in requests:
            record["locality"] = {"neighbors": {"prompt": record["locality_prompt"], "ground_truth": record["locality_ground_truth"]}}
            
    elif ds_name == "zsre_mend_eval":
        with open(args.ds_dir) as f:
            ds = json.load(f)
        requests = []
        for record in ds[:sample]:
            if len(record["alt"])>0:
                subject = record["subject"]
                prompt = record["src"]
                rephrase_prompt = record["rephrase"]
                target_new = record["alt"]
                ground_truth = record["answers"][0]
                locality = {"neighbors": {"prompt": record["loc"][len("nq question: "):], "ground_truth": record["loc_ans"]}}
                if subject in prompt:
                    requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, "rephrase_prompt":rephrase_prompt, 
                                    "locality":locality, "subject":subject})


    #chunk size only used for deciding whether to use consecutive editing
    if consecutive:
        chunk_size = len(requests)
    else:
        chunk_size = hparams.batch_size
        eval_interval = None


    print("update_weight: {} batch_size:{} number of edits: {}".format(hparams.mom2_update_weight, hparams.batch_size, len(requests)))
    print("consecutive:{} ds:{}".format(consecutive, ds_name))





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
            # print(outputs.shape, new_outputs.shape)
            return outputs
        return hook


    def validated_hook_print(new_module:torch.nn.Module, alpha:torch.Tensor):
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
            # print(outputs.shape, new_outputs.shape)
            return outputs
        return hook



    all_metrics, final_metrics = [], []
    for r,chunk in enumerate(_chunks(requests, chunk_size)):
        for i,record in enumerate(chunk):
            all_metrics.append({"case_id":i+r*chunk_size})
            all_metrics[i+r*chunk_size]["pre"] = compute_edit_quality(model, model_name, hparams, tokenizer, record, device=hparams.device)
            
        if not wohk:
            accu_params_set = apply_CoachHooK_to_model(model, tokenizer, chunk, hparams, alpha_z=alpha_z, eval_interval=eval_interval)
        else:
            _, weights_copy, accu_params_set = apply_CoachHooK_wohk_to_model(model, tokenizer, chunk, hparams, eval_interval=eval_interval)

        for n,accu_params in enumerate(accu_params_set):
            if not wohk:
                hook_handles = []          
                for i,(m_name,params) in enumerate(accu_params.items()):
                    # print(params["new_weight"].weight)
                    module = nethook.get_module(model, m_name)
                    if i==len(accu_params)-1:
                        # print(m_name)
                        handle = module.register_forward_hook(validated_hook_print(params["new_weight"], params["alpha"],))
                    else:
                        handle = module.register_forward_hook(validated_hook(params["new_weight"], params["alpha"]))
                    hook_handles.append(handle)
            else:
                for i,(w_name,params) in enumerate(accu_params.items()):
                    w = nethook.get_parameter(model, w_name)
                    w[...] = params["w"].to(w.device)
            
            if eval_interval is not None:
                #intermediate evaluation
                edit_num = (n+1)*eval_interval
                inter_metrics = all_metrics[:edit_num]
                for i,record in enumerate(requests[:edit_num]):
                    inter_metrics[i]["post"] = compute_edit_quality(model, model_name, hparams, tokenizer, record, device=hparams.device)
                final_metrics.append(inter_metrics)
            else:
                if n == len(accu_params_set)-1:
                    #only do final evaluation
                    for i,record in enumerate(chunk):
                        all_metrics[i+r*chunk_size]["post"] = compute_edit_quality(model, model_name, hparams, tokenizer, record, device=hparams.device)
            
            if not wohk:
                for hook_handle in hook_handles:
                    hook_handle.remove()
            else:
                for w_name,w in weights_copy.items():
                    w_e = nethook.get_parameter(model, w_name)
                    w_e[...] = w.to(w_e.device)     
        
            
            
            
    if eval_interval is None:
        final_metrics.append(all_metrics)


    all_res = {}
    for i,inter_metrics in enumerate(final_metrics):
        rewrite_acc, rephrase_acc = 0, 0
        locality, portability = {}, {}
        for metric in inter_metrics:
            post_metric = metric["post"]
            pre_metric = metric["pre"]
            rewrite_acc += post_metric["rewrite_acc"][0]
            
            if "rephrase_acc" in post_metric.keys():
                rephrase_acc += post_metric["rephrase_acc"][0]
            
            if "locality" in post_metric.keys():
                for key,value in post_metric["locality"].items():
                    if key not in locality.keys():
                        locality[key] = 0
                    # locality[key] += np.mean(np.equal(pre_metric["locality"][key], value)).item()
                    if "_output" in key:
                        locality[key] += np.mean(np.equal(metric["pre"]["locality"][key], value)).item()
                    elif "_acc" in key:
                        locality[key] += value[0]
                    
            if "portability" in post_metric.keys():
                for key,value in post_metric["portability"].items():
                    if key not in portability.keys():
                        portability[key] = 0
                    portability[key] += value[0]
            
        print(inter_metrics)
        results = {"rewrite_acc": rewrite_acc/len(inter_metrics), "rephrase_acc": rephrase_acc/len(inter_metrics)}
        for k,v in locality.items():
            locality[k] = v/len(inter_metrics)
        for k,v in portability.items():
            portability[k] = v/len(inter_metrics)
        results["locality"] = locality
        results["portability"] = portability
        print(results)
        all_res[i] = results
        
        
        
    run_config = {
                "model_name": model_name,
                "editing model": editing_method,
                "batch_size": hparams.batch_size,
                "consecutive": consecutive,
                "mom2_update_weight": hparams.mom2_update_weight,
                "dataset": ds_name,
                "requests_size": len(requests),
                "eval_interval": eval_interval if eval_interval is not None else "no",
                "eval_instance_num": [len(inter_metrics) for inter_metrics in final_metrics],
                "alpha_z": alpha_z,
                "wohk": wohk,
                "editing layers": hparams.layers,
            }

    print(run_config)
    print(all_res)



if __name__=="__main__":
    main()