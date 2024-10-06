gpu=0
wandb_mode=disabled

# arch related
arch=resnet18
weight_decay=0.0005
lr=0.001
blr=0
min_lr=0
warmup_epochs=0

# data
data_root=./data
dataset=TerraIncognita

workers=0
val_workers=0
num_epochs=20
batch_size=16
iters_per_epoch=0
num_rounds=200

performative_temperature=0.1
init_dirichlet_alpha=100
base_size=1000
test_base_size=1000
shift_type=class

pretraining_for_predictors=False

prior_predictor=False
oracle=False
no_training=True

train_prior_predictor=True
full_covariate_shift=True

continue_check=True

check_path=./performative_prediction/checkpoint_dataset_TerraIncognita_arch_resnet18_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_fnvnnbug/resnet18_TerraIncognita_temp_0.8_round_25_final.pth
prior_path=./performative_prediction/checkpoint_dataset_TerraIncognita_arch_resnet18_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_fnvnnbug/priors.pth


print_freq=10
eval_freq=5

exp_name=round_${num_rounds}_cov_shift_with_prior_pred_training


declare -a seed=(0 7 77)


# naming
group=checkpoint_dataset_${dataset}_arch_${arch}_shift_type_${shift_type}_temperature_${performative_temperature[k]}_init_alpha_${init_dirichlet_alpha[l]}_base_size_${base_size}
export CUDA_VISIBLE_DEVICES=${gpu} 
export WANDB_MODE=${wandb_mode}

for ((i=0;i<${#seed[@]};++i)); do
    python ./main.py \
    --seed ${seed[i]} \
    --exp_group ${group} --exp_name ${exp_name} \
    --data ${dataset} --arch ${arch} --data_root ${data_root} --workers ${workers} --val_workers ${val_workers} \
    --lr ${lr} --blr ${blr} --iters_per_epoch ${iters_per_epoch} --batch_size ${batch_size} --weight_decay ${weight_decay} \
    --epochs ${num_epochs} --eval_freq ${eval_freq} \
    --num_rounds ${num_rounds} \
    --performative_temperature ${performative_temperature} \
    --init_dirichlet_alpha ${init_dirichlet_alpha[l]} \
    --base_size ${base_size} --test_base_size ${test_base_size} --shift_type ${shift_type} \
    --warmup_epochs ${warmup_epochs} --print_freq ${print_freq} \
    --prior_predictor ${prior_predictor} --prior_path ${prior_path} --no_training ${no_training} \
    --continue_check ${continue_check} --check_path ${check_path} --oracle ${oracle} --pretraining_for_predictors ${pretraining_for_predictors} \
    --full_covariate_shift ${full_covariate_shift} --train_prior_predictor ${train_prior_predictor}
done