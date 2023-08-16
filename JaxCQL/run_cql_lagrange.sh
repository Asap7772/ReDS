envs=(antmaze-medium-diverse-v0 antmaze-medium-play-v0)
cql_target_action_gaps=(0.8)
seeds=(42)
num_exps=0
gpus=(0 1 2 3 4 5 6 7)

for seed in "${seeds[@]}"; do
for env in "${envs[@]}"; do
for a in "${cql_target_action_gaps[@]}"; do
    which_gpu=$((num_exps % ${#gpus[@]}))
    gpu=${gpus[$which_gpu]}
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "env: $env, seed: $seed, tau: $t, alpha: $a, gpu: $gpu"
    command="XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main \
    --env $env \
    --logging.output_dir './experiment_output' \
    --logging.online True \
    --logging.project 'CQL_antmaze_arch' \
    --seed $seed \
    --cql.cql_lagrange True \
    --cql.cql_min_q_weight=5.0 \
    --cql.cql_target_action_gap=0.8 \
    --policy_arch=256-256 \
    --qf_arch=256-256-256-256 \
    --eval_n_trajs=20 \
    --reward_scale=10.0 \
    --reward_bias=-5"

    echo $command
    eval $command &

    sleep 30 # wait for the experiment to start
    num_exps=$((num_exps+1))
done
done
done

echo "num_exps: $num_exps"
