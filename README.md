# QDFM: Flow Matching for Offline Reinforcement Learning with Discrete Actions

Official PyTorch implementation of **Flow Matching for Offline Reinforcement Learning with Discrete Actions**, accepted at **NeurIPS 2026**.

**Paper:** https://arxiv.org/abs/2602.06138

This repository provides Q-weighted Discrete Flow Matching (QDFM) implementations and experiment scripts for discrete-action offline reinforcement learning, including single-objective, multi-objective, and multi-agent settings.

## Requirements

Python 3.8 or later with the following packages

    torch numpy matplotlib gymnasium

Additional packages for specific experiments

    minari mujoco         (for discretized MuJoCo)
    mo_gymnasium          (for multi objective discrete benchmarks)
    pysc2 smac            (for SMAC multi agent experiments)

Install the base requirements with pip

    pip install torch numpy matplotlib gymnasium

## File Structure

    bench/              Shared networks (MLP, RateNet, DiscreteQ, CategoricalPolicy, VectorQ, QMIX)
    algos/              All algorithm implementations
        qdfm.py         Single objective QDFM
        qdfm_mo.py      Multi objective QDFM with preference conditioning
        cql.py          Conservative Q Learning
        cql_mo.py       Multi objective scalarized CQL
        iql.py          Implicit Q Learning
        awac.py         Advantage Weighted Actor Critic
        bcq.py          Batch Constrained Q Learning
        brac.py         Behavior Regularized Actor Critic
        greedyq.py      Greedy Q (argmax baseline)
    envs/               Custom environments
        fork_env.py     Multi Objective Fork gridworld
        matrix_game.py  Two agent coordination game
    utils.py            Seed setting
    discretize.py       Action discretization via k means
    eval_policy.py      Policy evaluation for continuous control

## Experiment Scripts

### Multi Goal Gridworld (Section 5, Appendix B.19)

    python run_multigoal.py --method qdfm --K 4 --seed 42 --device cuda --out results_multigoal

Methods available are qdfm cql iql awac bcq brac greedyq.
K values available are 2 3 4 5.
Run all 140 combinations (7 methods x 4 K values x 5 seeds) then aggregate

    python aggregate_multigoal.py --dir results_multigoal --fig_dir figures

### Discretized MuJoCo (Section 5.1)

    python run_mujoco.py --task hopper_medium --method qdfm --seed 42 --K_act 256 --device cuda --out results_mujoco

Tasks available are halfcheetah_medium halfcheetah_expert hopper_medium hopper_expert walker2d_medium walker2d_expert.

### Multi Objective Discrete Benchmarks (Section 5.2)

Requires mo_gymnasium

    python run_mo_discrete.py --env deep_sea_treasure --method qdfm --seed 42 --device cuda
    python run_mo_discrete.py --env resource_gathering --method cql --seed 42 --device cuda

### Multi Objective Fork (Appendix B.13)

    python run_fork.py --seed 42 --device cuda

### CartPole Two Objective (Appendix B.12)

    python run_cartpole_mo.py --seed 42 --device cuda

### CTMC Sampler Correctness (Appendix B.11)

    python run_ctmc_check.py --env CartPole-v1 --device cpu
    python run_ctmc_check.py --env Acrobot-v1 --device cpu

### Two Agent Matrix Game (Appendix B.16)

    python run_matrix_game.py --seed 42 --device cuda

### SMAC Multi Agent (Section 5.3)

Requires smac and StarCraft II installation. Provide a preprocessed data file in numpy format

    python run_smac.py --map 2s3z --method cql --seed 42 --data_path data/smac_2s3z.npy --device cuda

## Seeds

All reported results use seeds 42, 123, 456, 789, and 2024.
