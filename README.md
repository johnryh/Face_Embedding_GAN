# Mask Embedding in conditional GAN for Guided Synthesis of High Resolution Images

Yinhao Ren, Zhe Zhu, Yingzhou Li, Joseph Lo

## Abstract
Recent advancements in conditional Generative Adversarial Networks
(cGANs) have shown promises in label guided image synthesis. Semantic
masks, such as sketches and label maps, are another intuitive and
effective form of guidance in image synthesis. Directly incorporating
the semantic masks as constraints dramatically reduces the variability
and quality of the synthesized results. We observe this is caused by
the incompatibility of features  from different  inputs (such as mask
image and latent vector) of the generator. To use semantic masks as
guidance whilst providing realistic synthesized results with fine
details, we propose to use mask embedding mechanism  to allow for
a more efficient  initial feature projection  in the generator. We
validate the effectiveness of our approach by training a mask guided
face generator using  CELEBA-HQ dataset. We can generate realistic and
high resolution facial images up to the resolution of 512 by 512
with a mask guidance.

![](teasers/teaser.png)


## Training
First setup the dataset into appropriate format and than modified the `config.py` file to specify the path to training data. The input pipeline is build in the `Input_Pipeline_celeba.py` using tesnorflow's Dataset API. Network impementations are in `network_utility.py`. 

To train the model for `Phase N` call `train` with all required parameters:

`python3 -W ignore train.py --GPU NUM_GPUs --phase PHASE_NUM --smooth USE_SMOOTH --size SIZE --epoch TOTAL_EPOCH --batch_size BATCH_SIZE --lr LEARNING_RATE --n_critic NUM_CRITIC --use_embedding USE_EMBEDDING`

You can also use provided bash script `starts_jobs_embedding.sh` to execuate each phase of the progressive training schedule:

`./starts_jobs_embedding.sh`
