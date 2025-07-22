# Typical setup routine (TODO: pack into Dockerfile)

```bash
sudo apt-get update
sudo apt-get install fd-find
ln -s $(which fdfind) ~/.local/bin/fd
sudo apt-get install autojump
```

## Configure aliases

```bash
echo '' >> ~/.bashrc
echo 'source /usr/share/autojump/autojump.sh' >> ~/.bashrc
echo "alias ..='cd ..'" >> ~/.bashrc
echo "alias ...='cd ../..'" >> ~/.bashrc
echo "alias x='exit'" >> ~/.bashrc
echo "alias lt='ls -lrth'" >> ~/.bashrc
echo "alias la='ls -A'" >> ~/.bashrc
echo "alias vim='vim -p'" >> ~/.bashrc
```

## Configure vim

```bash
# First,
# Add vim-flake8 to .vimrc
# Plugin 'https://github.com/nvie/vim-flake8'
# Then,
vim +PluginInstall
```
 
## Install gym and dywa

```bash
python3 -m pip install -e /opt/isaacgym/python
python3 -m pip install -e /home/user/DyWA/dywa
```
