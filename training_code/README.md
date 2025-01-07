The list of models are as follows 
- **ResNet** (50 layers, 101 layers, 152 layers)
- **EfficientNet** (b0~b8, l2 size models)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

   

# The example of pre-training model to predict current weight status 
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 1e-3 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 4 --exp_name pretrain_BMI_partion0 --num_target BMI_sds_baseline --resize 128 128 128  --study_sample ABCD_1y_after_become_overweight_pretrain_MDD  --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir 
```


# The example of fine-tuning model to predict future obesity risk

## full fine-tuning pre-trained model parameters
set ```--conv_unfreeze_iter``` as ```0```
### prediction of future overweight/obesity risk within 1 year
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name unfrozen_become_overweight_1year_partition0 --cat_target become_overewight  --resize 128 128 128  --study_sample ABCD_1y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 0
```
### prediction of future overweight/obesity risk within 2 years
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name unfrozen_become_overweight_2year_partition0 --cat_target become_overewight  --resize 128 128 128  --study_sample ABCD_2y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 0
```


## fine-tuning only the last linear projection layer of model while freezing all of the convolution layers
set ```--conv_unfreeze_iter``` larger than ```--epoch```
### prediction of future overweight/obesity risk within 1 year
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name frozen_become_overweight_1year_partition0 --cat_target become_overewight  --resize 128 128 128  --study_sample ABCD_1y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 99999
```
### prediction of future overweight/obesity risk within 2 years
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name frozen_become_overweight_2year_partition0 --cat_target become_overewight  --resize 128 128 128  --study_sample ABCD_2y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 99999
```  


# The example of fine-tuning model to predict the diagnoses of obesity-related psychiatric disorders (e.g., Major Depressive Disorder)
## full fine-tuning pre-trained model parameters (with undersampling case/control)
set ```--conv_unfreeze_iter``` as ```0```
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name unfrozen_MDD_parent_past_partition0 --cat_target MDD_parent_past  --resize 128 128 128  --study_sample ABCD_1y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 0
```

## fine-tuning only the last linear projection layer of model while freezing all of the convolution layers (with undersampling case/control)
set ```--conv_unfreeze_iter``` larger than ```--epoch```
```
python3 run_3DCNN_hard_parameter_sharing.py --optim AdamW  --lr 2e-4 --epoch 50 --model densenet3D121 --batch_size 32 --accumulation_steps 1 --exp_name frozen_MDD_parent_past_partition0 --cat_target MDD_parent_past  --resize 128 128 128  --study_sample ABCD_1y_after_become_overweight_pretrain_MDD --partitioned_dataset_number 0 --weight_decay 1e-4 --warmup_epoch 0 --checkpoint_dir {/path/to/ckpt_file} --finetune_undersample --conv_unfreeze_iter 99999
```

