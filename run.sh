python -m src.deepq.search --exp_name atari_deepq12 --distributional True --num_atoms 51 --double_q True --dueling True
python -m src.deepq.search --exp_name atari_deepq13 --double_q False --dueling True
python -m src.deepq.search --exp_name atari_deepq14 --double_q True --dueling True --n_step 3
python -m src.deepq.search --exp_name atari_deepq15 --double_q True --dueling True

python -m src.deepq.search --exp_name atari_deepq16 --double_q True --dueling True --noisy True
python -m src.deepq.search --exp_name atari_deepq17 --double_q True --dueling False

