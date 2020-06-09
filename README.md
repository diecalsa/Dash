# Manifold Learning Dashboard


## Table of content

- [Prerequisites](#prerequisites)
- [Project config](#project-config)
  - [Virtualenv creation](#virtualenv-creation)
  - [Installation](#installation)
  - [Environments](#environments)
  - [Github hooks](#github-hooks)
  - [Jupyter notebooks](#jupyter-notebooks)
- [Load tests](#load-tests)
  - [Installation](#installation)
  - [Run tests](#run-tests)
- [Coverage and Sonarqube](#coverage-and-sonarqube)
  - [Run Coverage](#run-coverage)
  - [Run Sonarqube](#run-sonarqube)
- [Websockets](#websockets)
  - [Run in local mode](#run-in-local-mode)

----

## Prerequisites

The prerequisites are:

- Git
- Python3
- pip3

Verify if Python / Pip has version 3 active:

```
$> python3 --version
$> pip3 --version
```

----

## Project config

### **Virtualenv creation**

Virtualenv configuration in localhost. We need to install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). It's compatible with BASH y ZSH.

```
$> pip3 install virtualenvwrapper
```

Open and edit  shell config file (~/.zshrc or ~/.bashrc). At the end of file, write this:

```
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

And close/open the terminal.

Quickstart:

```
$> which python3 #Output: /usr/bin/python3
$> mkvirtualenv --python=/usr/bin/python3 itaca
$> workon itaca
```

Now, the virtualenv with Python3 is installed and activated in your PC.

### **Installation**

Firstly, clone the repository:
```
$> git clone https://github.com/diecalsa/Dash.git
$> cd Dash
```

Install dependencies:
```
$> pip3 install -r requirements.txt
```

Run the project (local environment):
```
$> python3 newInterface.py
```

```


## Diego y Salim
