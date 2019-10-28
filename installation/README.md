# Installation

This document shows the steps to install the necessary tools (Python 3.X,
PyTorch, IPython, and VSCode) on Ubuntu 18.04, in the decreasing level of
importance.


# Ubuntu on Virtual Machine

You can use Ubuntu on a virtual machine.  The steps shown later in this
document were performed on a fresh installation of Ubuntu 18.04 (LTS) from an
ISO image downloaded from https://ubuntu.com/download/desktop.

You can follow this guide to install Ubuntu in VirtualBox:
https://itsfoss.com/install-linux-in-virtualbox/.

<!---
I've additionally installed VirtualBox Guest Additions, which enable some nice
features:
* the guest screen takes the entire place dedicated to it
* bidirectional copy and paste
-->


# Python

Python 3.6 should be pre-installed on Ubuntu installed from the ISO image
mentioned above (at least if you chose "normal" installation, I'm not sure
about the minimal installation).  Open terminal and type `python3` to verify
this.


# PyTorch

You can use the following commands to install PyTorch (without CUDA support):

    sudo apt install python3-pip
    pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

<!---
This command is proposed at https://pytorch.org/ if you choose `pip` and no
`cuda` support.
-->

Then, you can start a `python3` session and type:
```python
>>> import torch
```
to verify that it works.


# IPython

To install IPython, just type:

    pip3 install ipython

This will probably install `ipython` under `~/.local/bin/iptython` by default.
You can add `~/.local/bin` to your path by adding the following to your `~/.profile`
(if it's not already there):

```bash
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
```

After restart (or reloading the `~/.profile`) you should be able then to start
IPython by simply typing `ipython` in the terminal.


# VSCode

You can follow the steps from this following post to install VSCode:
https://linuxize.com/post/how-to-install-visual-studio-code-on-ubuntu-18-04/.

<!---
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
-->

Once you are done, you can run VSCode with `code`.  A pop-up message will show
up asking if you want to install the 'Python' extension, which you should
do.

It is also recommended to install the `flake8` linter, which works well with
PyTorch.
