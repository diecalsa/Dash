# Manifold Learning Dashboard


## Table of content

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Run the app](#run-the-app)
- [About the app](#about-the-app)
  - [Installation](#Built-with)
  - [Authors](#Authors)


----

## Prerequisites 

The prerequisites are:

- Git
- Python3
- pip

Verify if Python / Pip has version 3 active :snake: :

```
$> python3 --version
$> pip --version
```

----

## Getting Started


### **Create a virtual environment**

If you are running the app locally you might want to set up a virtual environment with conda or venv.

```
$> conda create --name <venv_name> python=3.8

# Windows
$> activate <venv_name>

#Linux, maxOS
$> source activate <venv_name>
```


### **Installation**

Clone the git repository:
```
$> git clone https://github.com/diecalsa/Dash.git
$> cd Dash
```

Install dependencies:
```
$> pip install -r requirements.txt
```

### **Run the app**

Run the project (local environment):
```
$> python app.py
```

## About the app

The app allows the user to reduce the datasets dimensionality within linear and non-linear algorithms such as PCA, t-SNE, IsoMAP, etc.
It's possible to:

#### Load your own CSV dataset
![Upload](https://github.com/diecalsa/Dash/blob/develop/src/upload_data.gif)

#### Visualize the data in 2D or 3D 
![General](https://github.com/diecalsa/Dash/blob/master/src/General.png)

#### Colorize by an original variable
#### Select the hover data to display (principal components + original features)
#### Choose to display only outliers by selecting the minimal distance to the origin
#### Download the new dataset consisting of the original features and the new ones (principal components)
![Download](https://github.com/diecalsa/Dash/blob/develop/src/download_data.gif)




## Built with

* [Dash](https://dash.plotly.com/) -
* [Plotly](https://plotly.com/)
* [Scikit-Learn](https://scikit-learn.org/stable/)


## Authors

* **[Salim Chikh](https://www.linkedin.com/in/salim-chikh-48b679109/)** - 
* **[Diego Calvete](https://www.linkedin.com/in/diego-calvete-010532b5/)** -
