# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:34:31 2024

@author: Li_Sh
"""

import os
import torch
from copy import deepcopy
from EasyEdit.easyeditor.util import nethook
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from CoachHooK_main import apply_CoachHooK_to_model, MEMITHyperParams
import pickle
import argparse
import json
from EasyEdit.easyeditor.models.rome.repr_tools import get_words_idxs_in_templates


# base_dir = "/projdata11/info_fil/sli/TPR_model_editing"
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def main():
    parser = argparse.ArgumentParser(description='CoachHook editing scope investigation')
    parser.add_argument('--stats_dir', type=str, default="./data/stats", help='the path of stat file')
    parser.add_argument('--model', type=str, choices=["EleutherAI/gpt-j-6B","gpt2-xl"], default="gpt2-xl", help='the model to use')
    parser.add_argument("--ds_dir", type=str, choices=["./Editing_data/zsre/zsre_mend_eval.json", "./Editing_data/counterfact/counterfact-edit.json"], 
                        default="./editing-data/data/zsre/zsre_mend_eval.json", help="the dataset directory")
    parser.add_argument('--consecutive', type=int, choices=[0,1], default=0, help='whether the editing happens in a consecutive way')
    parser.add_argument('--bsz', type=int, default=30, help='the editing batch size to use')
    parser.add_argument('--mom2_update_weight', type=int, default=15000, help='the moment2 update weight')
    parser.add_argument('--alpha_z', type=float, default=2.2, help='the initial alpha')
    parser.add_argument('--sample', type=int, default=100, help='number of samples to use')
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
    # print(model)


    hparams.stats_dir = args.stats_dir
    hparams.batch_size = args.bsz
    consecutive = True if args.consecutive else False
    hparams.mom2_update_weight = args.mom2_update_weight
    alpha_z = args.alpha_z
    eval_interval = 20*hparams.batch_size
    if "counterfact-edit" in args.ds_dir:
        ds_name = "counterfact-edit"
    else:
        ds_name = "zsre_mend_eval"
    mode = ""
    sample = args.sample
    r_record, g_record = [], []




    if ds_name == "counterfact-edit":
        with open(args.ds_dir) as f:
            ds = json.load(f)
        requests = ds[:sample]

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
                locality = {"neighbors": {"prompt": record["loc"], "ground_truth": record["loc_ans"]}}
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
    # print("alpha_diff:{} alpha_accu:{} or_and:{}".format(alpha_diff, alpha_accu, "no; only alpha_diff"))
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


    def validated_hook_recording(new_module:torch.nn.Module, alpha:torch.Tensor):
        def hook(module, inputs, outputs):
            # print(inputs, outputs)
            x = inputs[0]
            # print(x.shape)
            new_module.to(device=x.device, dtype=x.dtype)
            new_outputs = new_module(x)
            
            diff = new_outputs - outputs
            norm_diff = diff.norm(dim=-1)
            
            z_scores = (norm_diff - norm_diff.mean(dim=-1,keepdim=True)) / norm_diff.std(dim=-1,keepdim=True)
            
            if mode == "reliability":
                r_record.append(z_scores.detach().cpu().squeeze().tolist())
            elif mode == "generality":
                g_record.append(z_scores.detach().cpu().squeeze().tolist())
            
            idx_bool = torch.ge(z_scores, alpha)
            
            outputs[idx_bool,:] = new_outputs[idx_bool,:]
            # print(outputs.shape, new_outputs.shape)
            return outputs
        return hook


    accu_params_set = apply_CoachHooK_to_model(model, tokenizer, requests, hparams, alpha_z=alpha_z, eval_interval=eval_interval)
    accu_params = accu_params_set[-1]



    records = deepcopy(requests)
    for i, record in enumerate(records):
        if record["target_new"][0] != " ":
            # Space required for correct tokenization
            records[i]["target_new"] = " " + record["target_new"]
        
        if '{}' not in record['prompt']:
            assert record['subject'] in record['prompt'] or \
                    print(f"Subject:{record['subject']} do not exist in prompt: {record['prompt']}")
        
            records[i]['prompt'] = records[i]['prompt'].replace(records[i]['subject'], '{}')
        
        if '{}' not in record['rephrase_prompt']:
            assert record['subject'] in record['rephrase_prompt'] or \
                    print(f"Subject:{record['subject']} do not exist in prompt: {record['rephrase_prompt']}")
        
            records[i]['rephrase_prompt'] = records[i]['rephrase_prompt'].replace(records[i]['subject'], '{}')

            





    idxs_r = get_words_idxs_in_templates(tokenizer, 
                                        context_templates=[record["prompt"] for record in records], 
                                        words=[record["subject"] for record in records],
                                        subtoken=hparams.fact_token[len("subject_"):])

    idxs_g = get_words_idxs_in_templates(tokenizer, 
                                        context_templates=[record["rephrase_prompt"] for record in records], 
                                        words=[record["subject"] for record in records],
                                        subtoken=hparams.fact_token[len("subject_"):])    


    for i,(m_name,params) in enumerate(accu_params.items()):
        module = nethook.get_module(model, m_name)
        if i==len(accu_params)-1:
            # print(m_name)
            handle = module.register_forward_hook(validated_hook_recording(params["new_weight"], params["alpha"],))
        else:
            handle = module.register_forward_hook(validated_hook(params["new_weight"], params["alpha"]))

    for request in requests:
        for key in ["prompt", "rephrase_prompt"]:
            inputs = tokenizer(request[key], return_tensors="pt", padding=True).to(model.device)
            if key == "prompt":
                mode = "reliability"
            elif key == "rephrase_prompt":
                mode = "generality"
            model(**inputs)


    print(idxs_r, idxs_g)
    print("r_record", r_record)
    print("g_record", g_record)
    print(params["C_accu"], params["C_accu"].shape)
    print(params["new_weight"].weight.data, params["new_weight"].weight.data.shape)
    print(params)
        
        

    with open('r_idxs_{}.pkl'.format(sample), 'wb') as f:
        pickle.dump(idxs_r, f)
    with open('r_record_{}.pkl'.format(sample), 'wb') as f:
        pickle.dump(r_record, f)

    with open('g_idxs_{}.pkl'.format(sample), 'wb') as f:
        pickle.dump(idxs_g, f)
    with open('g_record_{}.pkl'.format(sample), 'wb') as f:
        pickle.dump(g_record, f) 

    # params["C_accu"] = params["C_accu"].tolist()
    # params["new_weight"] = params["new_weight"].weight.data.tolist()
    with open("params_{}.json".format(sample), "w") as f:
        json.dump({"alpha":params["alpha"]}, f)



if __name__=="__main__":
    main()