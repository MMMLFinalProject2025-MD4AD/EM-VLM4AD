#! /bin/bash -f

#pre-train
CUDA_VISIBLE_DEVICES=0 python ./train.py --batch-size 8 --epochs 8 --freeze-lm --num-workers 16 --load-checkpoint --checkpoint-file ./multi_frame_results/T5-Base/latest_model.pth --output-dir /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/ --lm T5-Base --load-orig-format
CUDA_VISIBLE_DEVICES=0 python ./train.py --batch-size 8 --epochs 8 --freeze-lm --num-workers 16 --load-checkpoint --checkpoint-file ./multi_frame_results/T5-Large-Q/latest_model.pth --output-dir /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/ --lm T5-Large --load-orig-format
CUDA_VISIBLE_DEVICES=0 python ./train.py --batch-size 8 --epochs 8 --freeze-lm --num-workers 16 --load-checkpoint --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/latest_model_saved.pth --output-dir /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/ --lm T5-Large


#train wo fine-tune
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/ --learning-rate 1e-5 --feat bevfusion
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/ --learning-rate 1e-5 --feat bevfusion
CUDA_VISIBLE_DEVICES=3 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/ --learning-rate 5e-4 --feat bevfusion

#train w/ fine-tune
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-4/ --learning-rate 1e-4 --feat bevfusion --restart
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-5/ --learning-rate 1e-5 --feat bevfusion --restart
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr5e-4/ --learning-rate 5e-4 --feat bevfusion --restart
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/latest_model_saved.pth --load-checkpoint  --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-Q-Large_lr1e-4/ --learning-rate 1e-4 --feat bevfusion --lm T5-Large --restart



#evaluation wo fine-tune
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/eval_result
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/eval_result
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-4/eval_result

#evaluation w fine-tune
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr5e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr5e-4/eval_result
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-4/eval_result
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-5/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-5/eval_result

