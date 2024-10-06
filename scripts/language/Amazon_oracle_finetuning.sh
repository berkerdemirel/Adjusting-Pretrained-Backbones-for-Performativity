gpu=0
wandb_mode=disabled

# arch related
arch=distilbert-base-uncased
weight_decay=1e-2
lr=1e-5
blr=0
min_lr=0
warmup_epochs=0 # typically 20% of the entire epochs

# data
data_root=./data/
dataset=Amazon
max_token_length=512

workers=0
val_workers=0
num_epochs=3
batch_size=24
iters_per_epoch=0
num_rounds=200



init_dirichlet_alpha=100
base_size=1000
test_base_size=2000
shift_type=class

prior_predictor=False
oracle=False
no_training=False

continue_check=True
pretraining_for_predictors=False

check_path=./performative_prediction/dataset_Amazon_arch_distilbert-base-uncased_shift_type_class_temperature_1_init_alpha_100_base_size_1000/seed_7/distilbert-base-uncased_Amazon_temp_1.0_round_20_final.pth
prior_path=./performative_prediction/dataset_Amazon_arch_distilbert-base-uncased_shift_type_class_temperature_1_init_alpha_100_base_size_1000/seed_7/priors.pth

print_freq=10
eval_freq=5

declare -a performative_temperature=(0.05 0.1 0.45)
declare -a seed=(0 7 77)

if [ "$prior_predictor" = "True" ]; then
    exp_name=round_${num_rounds}_prior_pred
    if [ "$oracle" = "True" ]; then
        exp_name=round_${num_rounds}_oracle
    fi
else
    exp_name=round_${num_rounds}_vanilla
fi


for ((k=0;k<${#performative_temperature[@]};++k)); do
group=checkpoint_dataset_${dataset}_arch_${arch}_shift_type_${shift_type}_temperature_${performative_temperature[k]}_init_alpha_${init_dirichlet_alpha[l]}_base_size_${base_size}
for ((i=0;i<${#seed[@]};++i)); do
    CUDA_VISIBLE_DEVICES=${gpu} WANDB_MODE=${wandb_mode} python ./main.py \
    --seed ${seed[i]} \
    --exp_group ${group} --exp_name ${exp_name} \
    --data ${dataset} --arch ${arch} --data_root ${data_root} --workers ${workers} --val_workers ${val_workers} \
    --lr ${lr} --blr ${blr} --iters_per_epoch ${iters_per_epoch} --batch_size ${batch_size} --weight_decay ${weight_decay} \
    --epochs ${num_epochs} --eval_freq ${eval_freq} \
    --num_rounds ${num_rounds} \
    --performative_temperature ${performative_temperature[k]} \
    --init_dirichlet_alpha ${init_dirichlet_alpha} \
    --base_size ${base_size} --test_base_size ${test_base_size} --shift_type ${shift_type}  \
    --warmup_epochs ${warmup_epochs} --print_freq ${print_freq} \
    --prior_predictor ${prior_predictor} --prior_path ${prior_path} --no_training ${no_training} \
    --continue_check ${continue_check} --check_path ${check_path} --oracle ${oracle} --pretraining_for_predictors ${pretraining_for_predictors}
done
done
