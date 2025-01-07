The list of models are as follows 
- **ResNet** (50 layers, 101 layers, 152 layers)
- **EfficientNet** (b0~b8, l2 size models)
- **DenseNet** (121 layers, 169 layers, 201 layers, 264 layers)

   

### The example of pre-training model to predict current weight status 

```
python3 run_3DCNN_hard_parameter_sharing.py --num_target BMI  --optim AdamW --lr 1e-3 --gpus 4 5 --exp_name BMI_pretraining --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32

```
or
```
python3 run_3DCNN_hard_parameter_sharing.py --cat_target sex  --optim SGD --lr 1e-3 --gpus 4 5 --exp_name sex_test --model {model_name} --epoch 300 --batch_size 32 --accumulation_steps 32
```
  


### Matching baseline year sample and 2 years after sample 
If you use option ```--matching_baseline_2years``` with checkpoint trained with baseline year  
