# SVD-NAS

official implementation of the WACV 2023 paper [SVD-NAS:Coupling Low-Rank Approximation and Neural Architecture Search](https://arxiv.org/abs/2208.10404) 

## Before starting
```
pip install -r requirement.text
```


## B1 NAS
the first iteration of NAS
```
python svd_nas_main.py \
--data <path of ImageNet folder> \
--gpu <device id> \
--multi_obj_beta <value of beta> \
--model_name <name of the model> \
--svd_option B1
```

## B2 NAS
the second iteration of NAS. fill the B1 config using the csv generated in the first iteration of NAS
```
python svd_nas_main.py \
--data <path of ImageNet folder> \
--gpu <device id> \
--multi_obj_beta <value of beta> \
--last_iteration_schemes <B1 config> \
--last_iteration_groups <B1 config> \
--last_iteration_ranks <B1 config> \
--model_name <name of the model> \
--svd_option B1
```

## Generate Synthesised Dataset
```
python distilled_data.py \
--gpu <device id> \
--model_name <name of the model>
```

## Knowledge Distillation
for B1 distillation, fill the B1 config using the csv generated in the first iteration of NAS
```
python kd_low_rank.py \
--data <path of ImageNet folder> \
--gpu <device id> \
--kd_data_src <path of Synthesised Dataset> \
-s <B1 config> \ 
-g <B1 config> \
-r <B1 config> \
--model_name <name of the model> \
--svd_option B1 
```
for B2 distillation, fill the B2 config using the csv generated in the second iteration of NAS, and fill the B1 config using the csv generated in the first iteration of NAS but replace the ranks with the output of "svd_nas_main.py#L193" because of the "withdraw" strategy.
```
python kd_low_rank.py \
--data <path of ImageNet folder> \
--gpu <device id> \
--kd_data_src <path of Synthesised Dataset> \
-s <B2 config> \ 
-g <B2 config> \
-r <B2 config> \
--last_iteration_schemes <B1 config> \
--last_iteration_groups <B1 config> \
--last_iteration_ranks <B1 config> \
--model_name <name of the model> \
--svd_option B2 
```
