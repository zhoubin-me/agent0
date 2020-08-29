# python -m agent0.deepq.run --game atari6 --algo all --exp_name atari_all --total_steps 20000000
python -m agent0.deepq.run --game atari6 --algo dqn --feature_mult 2 --exp_name atari_feature_mult2
python -m agent0.deepq.run --game atari6 --algo c51 --noisy --prioritize --n_step 3 --exp_name atari_rainbow4
python -m agent0.deepq.run --double_q --dueling --prioritize --noisy --algo c51 --n_step 3 --game atari6 --exp_name atari_rainbow
