apt-get -y install vim git htop tmux fish
git@github.com:zhoubin-me/agent0.git
cd agent0
pip install uv
uv python install 3.12
uv venv --python 3.12
uv sync