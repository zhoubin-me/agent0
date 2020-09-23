# python -m agent0.deepq.run --game atari6 --algo all --exp_name atari_all --total_steps 20000000
# python -m agent0.deepq.run --game atari6 --algo dqn --feature_mult 2 --exp_name atari_feature_mult2
# python -m agent0.deepq.run --game atari6 --algo c51 --noisy --prioritize --n_step 3 --exp_name atari_rainbow4
# python -m agent0.deepq.run --double_q --dueling --prioritize --noisy --algo c51 --n_step 3 --game atari6 --exp_name atari_rainbow

# python -m agent0.deepq.run --algo qr --policy soft_explore --game atari6 --exp_name atari_qr_soft_explore
# python -m agent0.deepq.run --algo gmm --policy soft_explore --game atari6 --exp_name atari_gmm_soft_explore

# python -m agent0.deepq.run --algo gmm --game atari6 --exp_name atari_gmm_gaussian_reward


python -m agent0.ddpg.run --algo ddpg --game mujoco7 --exp_name mujoco_ddpg
python -m agent0.ddpg.run --algo sac --game mujoco7 --exp_name mujoco_sac
python -m agent0.ddpg.run --algo td3 --game mujoco7 --exp_name mujoco_td3