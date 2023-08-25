import argparse, sys, time
sys.path.append('/home/zhhao/audioST/instruct_speech_llama')
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch, transformers
from instruct_speech_new.eval.utils import disable_torch_init
from instruct_speech_new.model.model import SpeechLlamaForCausalLM, SpeechLlamaConfig
from instruct_speech_new.model.utils import KeywordsStoppingCriteria
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from instruct_speech_new.train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
from instruct_speech_new import conversation as conversation_lib
from instruct_speech_new.conversation import SeparatorStyle

import os
import requests

import torch.nn.functional as F

prompt_list = ['Translate the following english sentence into German: ',]

def eval_model(args):
    load_type = torch.float16 # torch.float32
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
        #load_in_8bit=True,
        #device_map='auto',      
    ).cuda()    
    test_dataset = PromptSpeechToTextDatasetCreator.from_tsv(args.data_path, args.data_split)
  
    if not os.path.exists(os.path.join(args.result, args.data_split)):
        os.makedirs(os.path.join(args.result, args.data_split))
        
    ref_file = open(os.path.join(args.result, args.data_split, "ref"), "w")
    hyp_file = open(os.path.join(args.result, args.data_split, "hyp"), "w")
    conv = conversation_lib.conv_vicuna_v1_1.copy()
    for test_data in tqdm(test_dataset):
        src, ref, id = test_data.src_text, test_data.target, test_data.id                     
        qs = prompt_list[0] + src
        conv.messages = []
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_inputs = conv.get_prompt()
        inputs = tokenizer([prompt_inputs])
        input_ids = torch.as_tensor(inputs.input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)   
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids.cuda(),
                #do_sample=True,
                #temperature=0.8,
                num_beams=1,
                max_new_tokens=500,
                stopping_criteria=[stopping_criteria]) 
        input_token_len = input_ids.cuda().shape[1]   
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print(outputs)
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(f"{id} decode complete,\nref:{ref} \nhyp:{outputs}")
        print(f"{id}\t{ref}", file=ref_file)
        print(f"{id}\t{outputs}", file=hyp_file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    args = parser.parse_args()

    eval_model(args)
