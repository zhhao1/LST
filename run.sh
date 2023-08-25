# extral
cd docopt-0.6.2
python setup.py install
pip install num2words-0.5.12-py3-none-any.whl

# activate enviroment
conda activate py39
export PYTHONPATH=/home/zhhao/fairseq:$PYTHONPATH
cd /home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/train

# test vicuna-7b
python3 -m fastchat.serve.cli --model-path /home/zhhao/llm_model/vicuna-7b --load-8bit

# data prepare, stage1 and stage2
cd ./preprocess
python prep_librispeech_data.py --data-root /home/zhhao/data_source/SLR12/ --tgt-dir ../data/librispeech --mode train 
python prep_librispeech_data.py --data-root /home/zhhao/data_source/SLR12/ --tgt-dir ../data/librispeech --mode test
python prep_mustc_raw.py --data-root /home/zhhao/data_source/MUST-C/ --tgt-dir ../data/mustc --languages es  
CUDA_VISIBLE_DEVICES=1, python filter_tsv.py --dataset_name 'LIBRISPEECH' --tsv_root ../data/librispeech --asr_batch_size 36 \
                                             --asr_wer_threshold 0.0 --max_example_number 50000
CUDA_VISIBLE_DEVICES=1, python filter_tsv.py --dataset_name 'MUSTC' --tsv_root ../data/mustc/en-de --asr_batch_size 36 \
                                             --asr_wer_threshold 0.0 --max_example_number 50000      

# mt train
llm_model=/home/zhhao/llm_model/vicuna-7b
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-de/
save_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/mt/mustc/en-de/run1

torchrun --nnodes=4 --nproc_per_node=4 --master_port=12345 --node_rank=3 --master_addr="192.168.1.35"  \
    train_mt.py \
    --model_name_or_path ${llm_model} \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --output_dir ${save_path} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --seed 1234 \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json

# stage1 train
# activate enviroment

llm_model=/home/zhhao/llm_model/llama2/13b
ssl_model=/home/zhhao/ssl_model/20230627/wav2vec_vox_960h_pl.pt
#ssl_model=/home/zhhao/ssl_model/wav2vec_960/libri960_big.pt
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-es/
#data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/librispeech
save_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/en-es/stage1/run2

torchrun --nnodes=4 --nproc_per_node=4 --master_port=12345 --node_rank=0 --master_addr="192.168.1.35"  \
    stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation True \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs  6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 1234 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json

# stage2 train
llm_model=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/en-fr/stage1/run2
#llm_model=/home/zhhao/llm_model/vicuna-7b
#ssl_model=/home/zhhao/ssl_model/wav2vec_960/wav2vec_small.pt
ssl_model=/home/zhhao/ssl_model/20230627/wav2vec_vox_960h_pl.pt
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-fr/
save_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/en-fr/stage2/run2

torchrun --nnodes=4 --nproc_per_node=4 --master_port=12345 --node_rank=3 --master_addr="192.168.1.35" \
    stage2_large.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation True \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 1234 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json # deepspeed_config_stage2_offload.json deepspeed_config_stage3.json


# test single file
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/stage1/run6/checkpoint-5000
speech_file=/home/zhhao/data_source/SLR12/test-clean/1089/134686/1089-134686-0000.flac
CUDA_VISIBLE_DEVICES=1, python ./generate.py --model-name ${model_path} --speech-file ${speech_file}

# cli
cd ../server/
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/stage2/run7/checkpoint-2000
speech_file=/home/zhhao/data_source/SLR12/test-clean/1089/134686/1089-134686-0000.flac
CUDA_VISIBLE_DEVICES=0, python ./cli.py --model-path ${model_path} --load-8bit --speech-file ${speech_file}

How many speakers are there in this speech
what is the language of this speech

# test must-c mt result
cd ../eval/
lang=de
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/mt/mustc/en-de/run1/checkpoint-1000
#model_path=/home/zhhao/llm_model/llama2/7b-chat
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-${lang}
result=${model_path}/en-${lang}/result_beam1
CUDA_VISIBLE_DEVICES=0, python ./eval_mt.py --model-name ${model_path} --data-path ${data_path} --data-split 'tst-COMMON' --result ${result}
python ./compute_bleu.py ${result}/tst-COMMON

# mt
run1 mustc, en-de, 128, 2e-5, epoch 3, based on vicuna 7B,
# w/o finetune
llama  1,   en-de 7B beam1:
vicuna 1.1, en-de 7B beam1:22.98 beam4: 24.46  13B beam1:24.32 beam4:25.35
vicuna 1.3, en-de 7B beam1:23.35 beam4: 24.89  13B beam1:24.62 beam4:25.96
vicuna 1.5, en-de 7B beam1:19.91 beam4: 19.98  13B beam1:25.62 beam4:26.34  (7b have some questions)

# w/ finetune
7B, checkpoint-1000 beam1:30.64 beam4:32.22
    checkpoint-2000 beam1:30.44 beam4:31.87


# test dataset asr
cd ../eval/
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/stage1/run15
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/librispeech
result=${model_path}/result_beam4
CUDA_VISIBLE_DEVICES=1, python ./test_dataset_asr.py --model-name ${model_path} --data-path ${data_path} --data-split 'test-clean' --result ${result} 
python ./compute_wer.py ${result}/test-clean

# test dataset st
# split tsv

# 7b
cd ../eval
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/stage2/run22/checkpoint-1700
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-de
result=${model_path}/result_beam4
CUDA_VISIBLE_DEVICES=3, python ./test_dataset.py --model-name ${model_path} --data-path ${data_path} --data-split 'tst-COMMON' --result ${result} --beam 4

# 13b or large
model_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/checkpoints/stage2/run22/checkpoint-1700
data_path=/home/zhhao/audioST/instruct_speech_llama/instruct_speech_new/data/mustc/en-de

python ./extract_adapter.py \
  --model_name_or_path ${model_path} \
  --extracted_name 'mm_length_adapter' \
  --output ${model_path}/length_adapter.bin 
python ./extract_adapter.py \
  --model_name_or_path ${model_path} \
  --extracted_name 'mm_mlp_adapter' \
  --output ${model_path}/mlp_adapter.bin 
cd ../eval/   
result=${model_path}/result_beam4_20_50
CUDA_VISIBLE_DEVICES=0,1, python ./test_dataset_large.py \
                            --model-name ${model_path} \
                            --length-adapter-path ${model_path}/length_adapter.bin \
                            --mlp-adapter-path ${model_path}/mlp_adapter.bin \
                            --data-path ${data_path} \
                            --data-split 'tst-COMMON20_50' \
                            --result ${result} \
                            --beam 4
                            
python ./compute_bleu.py ${result}/tst-COMMON20_50




