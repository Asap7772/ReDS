envs=(halfcheetah-random-v2 halfcheetah-expert-v2 antmaze-medium-diverse-v0 antmaze-medium-play-v0)
tau=(0.1 1 10 20)
alphas=(1 5 10 20)
seeds=1
num_exps=0
gpus=(2 3 4 5 6 7)

for env in "${envs[@]}"; do
for t in "${tau[@]}"; do
for a in "${alphas[@]}"; do
for seed in $(seq 1 $seeds); do
    which_gpu=$((num_exps % ${#gpus[@]}))
    gpu=${gpus[$which_gpu]}
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "env: $env, seed: $seed, tau: $t, alpha: $a, gpu: $gpu"
    command="python -m JaxCQL.conservative_sac_reds_main \
        --env $env \
        --logging.output_dir './experiment_output' \
        --logging.online True \
        --logging.project 'ReDS_D4RL' \
        --seed $seed \
        --cql.reds_temp $t \
        --cql.cql_min_q_weight $a"
    echo $command
    eval $command &

    sleep 120 # wait for the experiment to start
    num_exps=$((num_exps+1))
done
done
done
done

echo "num_exps: $num_exps"
