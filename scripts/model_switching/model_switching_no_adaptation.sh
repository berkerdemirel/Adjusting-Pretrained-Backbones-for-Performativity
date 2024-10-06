gpu=0
wandb_mode=disabled

# arch related
arch=resnet18
arch_rounds="resnet18:0;resnet34:60;resnet50:120"
weight_decay=0.0005
lr=0.001
blr=0
min_lr=0
warmup_epochs=0

# data
data_root=./data
dataset=ImageNet100

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
train_prior_predictor=False
oracle=False
no_training=True

full_covariate_shift=False

continue_check=True

check_path="./performative_prediction/checkpoint_dataset_ImageNet100_arch_resnet18_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_766txoj3/resnet18_ImageNet100_temp_0.8_round_25_final.pth;./performative_prediction/checkpoint_dataset_ImageNet100_arch_resnet34_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_67xn2jxm/resnet34_ImageNet100_temp_0.8_round_25_final.pth;./performative_prediction/checkpoint_dataset_ImageNet100_arch_resnet50_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_r64k4t4s/resnet50_ImageNet100_temp_0.8_round_25_final.pth"
prior_path=./performative_prediction/checkpoint_dataset_ImageNet100_arch_resnet18_shift_type_class_temperature_0.8_init_alpha_100_base_size_1000/round_25_vanilla_766txoj3/priors.pth

prior_pred_check=./performative_prediction/checkpoint_dataset_ImageNet100_arch_resnet18_shift_type_class_temperature_0.1_init_alpha_100_base_size_1000/round_200_prior_pred_h3216st3/prior_pred_ImageNet100_temp_0.1_round_200_final.pth

# method related
# method=baseline
print_freq=10
eval_freq=5

if [ "$prior_predictor" = "True" ]; then
    exp_name=round_${num_rounds}_prior_pred
    if [ "$oracle" = "True" ]; then
        exp_name=round_${num_rounds}_oracle
    fi
else
    exp_name=round_${num_rounds}_vanilla
fi

declare -a seed=(0 7 77)


# naming
group=checkpoint_dataset_${dataset}_arch_${arch}_shift_type_${shift_type}_temperature_${performative_temperature[k]}_init_alpha_${init_dirichlet_alpha[l]}_base_size_${base_size}
export CUDA_VISIBLE_DEVICES=${gpu} 
export WANDB_MODE=${wandb_mode}

for ((i=0;i<${#seed[@]};++i)); do
    python ./main.py \
    --seed ${seed} \
    --exp_group ${group} --exp_name ${exp_name} \
    --data ${dataset} --arch ${arch} --data_root ${data_root} --workers ${workers} --val_workers ${val_workers} \
    --lr ${lr} --blr ${blr} --iters_per_epoch ${iters_per_epoch} --batch_size ${batch_size} --weight_decay ${weight_decay} \
    --epochs ${num_epochs} --eval_freq ${eval_freq} \
    --num_rounds ${num_rounds} \
    --performative_temperature ${performative_temperature} \
    --init_dirichlet_alpha ${init_dirichlet_alpha} \
    --base_size ${base_size} --test_base_size ${test_base_size} --shift_type ${shift_type} \
    --warmup_epochs ${warmup_epochs} --print_freq ${print_freq} \
    --prior_predictor ${prior_predictor} --prior_path ${prior_path} --no_training ${no_training} \
    --continue_check ${continue_check} --check_path ${check_path} --oracle ${oracle} --pretraining_for_predictors ${pretraining_for_predictors} \
    --full_covariate_shift ${full_covariate_shift}  --arch_rounds ${arch_rounds} --prior_pred_check ${prior_pred_check} --train_prior_predictor ${train_prior_predictor}
done
done
done
done