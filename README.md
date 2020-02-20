# Anytime Autoregressive Model



### Audio samples

In dir `audio`: audio samples generated with 0.0625/0.25/1.0 fractions of full code length by anytime sampling on **ordered** codes.

In dir `audio_baseline`: audio samples generated with 0.0625/0.25/1.0 fractions of full code length by anytime sampling on **unordered** codes.



### Image generation:



#####Training :

- Step 1: Pretrain VQ-VAE with full code length:

```shell
python vqvae.py --hidden-size latent-size --k codebook-size --dataset name-of-dataset --data-folder paht-to-dataset  --out-path path-to-model --pretrain

latent-size: latent code length
codebook-size: codebook size
name-of-dataset: mnist / cifar10 / celeba
path-to-dataset: path to the roots of dataset
path-to-model: path to save checkpoints
```



- Step 2: Train ordered VQ-VAE:

```shell
python vqvae.py --hidden-size latent-size --k codebook-size --dataset name-of-dataset --data-folder paht-to-dataset  --out-path path-to-model --restore-checkpoint path-to-checkpoint --lr learning-rate

latent-size: latent code length
codebook-size: codebook size
name-of-dataset: mnist / cifar10 / celeba
path-to-dataset: path to the roots of dataset
path-to-model: path to save checkpoints
path-to-checkpoint: the path of the best checkpoint in Step 1
learning-rate: learning rate (recommended:1e-3)
```



- Step 3: Train autoregressive model

```shell
python train_ar.py --task integer_sequence_modeling \
path-to-dumped-codes --vocab-size codebook-size --tokens-per-sample latent-size \
--ae-dataset name-of-dataset --ae-data-path path to the roots of dataset --ae-checkpoint path-to-checkpoint --ae-batch-size 512 \
--arch transformer_lm --dropout dropout-rate --attention-dropout dropout-rate --activation-dropout dropout-rate \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --weight-decay 0.1 --clip-norm 0.0 \
--lr 0.002 --lr-scheduler inverse_sqrt --warmup-updates 3000 --warmup-init-lr 1e-07 \
--max-sentences ar-batch-size \
--fp16 \
--max-update iterations \
--seed 2 \
--log-format json --log-interval 10000000 --no-epoch-checkpoints --no-last-checkpoints \
--save-dir path-to-model

path-to-dumped-codes: path to the dumped codes of VQ-VAE model (fasten training process)
dropout-rate: dropout rate
latent-size: latent code length
codebook-size: codebook size
name-of-dataset: mnist / cifar10 / celeba
path-to-dataset: path to the roots of dataset
path-to-model: path to save checkpoints
path-to-checkpoint: the path of the best checkpoint in Step 2
ar-batch-size: batch size of autorregressive model
iterations: training iterations
```



##### Genearte

```shell
python3 generate.py --n-samples number-of-samples --out-path paht-to-img \
--tokens-per-sample latent-size --vocab-size codebook-size --tokens-per-target code-num \
--ae-checkpoint path-to-ae --ae-batch-size 512 \
--ar-checkpoint path-to-ar --ar-batch-size batch-size

number-of-samples: number of samples to be generated
path-to-img: path to the generated samples
latent-size: latent code length
codebook-size: codebook size
code-num: number of codes used to generated (Anytime sampling!)
path-to-ae: path to the VQ-VAE checkpoint in Step 2
path-to-ar: path to the Transformer checkpoint in Step 3
batch-size: batch size for Transforer
```

