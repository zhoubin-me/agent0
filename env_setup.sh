sudo apt-get -y install vim git htop tmux fish libgl1-mesa-glx
pip3 install uv
echo "set-option -g default-shell /usr/bin/fish" >> $HOME/.tmux.conf
mkdir -p "$HOME/.config/fish"
echo "fish_add_path $HOME/.local/bin" >> $HOME/.config/fish/config.fish
fish -c "cd agent0; uv python install 3.12; uv venv --python 3.12; uv sync"
tmux
