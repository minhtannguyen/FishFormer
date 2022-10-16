Please follow the instruction at : https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md and https://github.com/pytorch/fairseq/blob/main/examples/scaling_nmt/README.md to reproduce the baselines' results for the task Machine Translation on IWSLT 2014 De-EN and WMT'14 EN-DE respectively.

To reproduce the 2-global-head FiSHformer on IWSLT'14, run:
```
CUDA_VISIBLE_DEVICES=7 fairseq-train \
    path/to/data \
    --save-dir /path/so/save\
    --arch hdp_transformer_iwslt_de_en_4head_2soft_qk --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-update 10000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --seed 1608\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-last-epochs 1
```

To reproduce the 8-global-head FiSHformer on WMT'14, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=56\
    $(which fairseq-train) \
    path/to/data \
    --save-dir path/to/save\
    --arch smgk_transformer_vaswani_wmt_en_de_big_8soft_qk --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16\
    --update-freq 32\
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    --eval-bleu-remove-bpe --max-epoch 45 --keep-last-epochs 10
```
