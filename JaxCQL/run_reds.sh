envs=(halfcheetah-random-v2 halfcheetah-expert-v2 walker2d-random-v2 walker2d-expert-v2 ant-random-v2 ant-expert-v2)
tau=(0.0001 0.1)
alphas=(1 5 10 20)
seeds=(42 24)
num_exps=0
gpus=(0 1 2 3 4 5 6 7)

for seed in "${seeds[@]}"; do
for env in "${envs[@]}"; do
for t in "${tau[@]}"; do
for a in "${alphas[@]}"; do
    which_gpu=$((num_exps % ${#gpus[@]}))
    gpu=${gpus[$which_gpu]}
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "env: $env, seed: $seed, tau: $t, alpha: $a, gpu: $gpu"
    command="XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_reds_main \
        --env $env \
        --logging.output_dir './experiment_output' \
        --logging.online True \
        --logging.project 'ReDS_D4RL_lse' \
        --seed $seed \
        --cql.reds_temp $t \
        --cql.cql_min_q_weight $a"
    echo $command
    eval $command &

    sleep 30 # wait for the experiment to start
    num_exps=$((num_exps+1))
done
done
done
done

echo "num_exps: $num_exps"
