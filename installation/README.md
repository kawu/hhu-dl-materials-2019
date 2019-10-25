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


# Python

Python 3.6 should be pre-installed on Ubuntu installed from the ISO image
mentioned above (at least if you chose "normal" installation, I'm not sure
about the minimal installation).  Open terminal and type `python3` to verify
this.


# PyTorch

You can use the following commands to install PyTorch:

  sudo apt install python3-pip
  pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 

<!---
This command is proposed at https://pytorch.org/ if you choose `pip` and no
`cuda` support.
-->
