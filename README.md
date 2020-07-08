# Manifold Learning Dashboard


## Table of content

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Create a virtual environment](#create-a-virtual-environment)
  - [Installation in local host](#installation-in-local-host)
  - [Installation with docker](#installation-with-docker)
  - [Run the app](#run-the-app)
- [About the app](#about-the-app)
  - [Built with](#Built-with)
  - [Authors](#Authors)


----

## Prerequisites 

The prerequisites are:

- Git
- Python3
- pip
- docker (only if you want to install the app using docker)

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


### **Installation in local host**

Clone the git repository:
```
$> git clone https://github.com/diecalsa/Dash.git
$> cd Dash
```

Install dependencies:
```
$> pip install -r requirements.txt
```

### **Installation with docker**

Run the docker container:

```
$> sudo docker run -p 8050:8050 mlia/manifoldsdash:latest
```

This command will download (if necessary) the docker container and run it. To interact with the app, you just need to open a web browser, i.e. google chrome, and insert:

**127.0.0.1:8050**

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


#### Set up the display options
* Colorize the data by an original feature
* Select the hover data to display 
* Choose to display only outliers by selecting the minimal distance to the origin

![Download](https://github.com/diecalsa/Dash/blob/develop/src/data_visualization.gif)

#### Explore the data
Your are able to visualize the data in 2D and 3D and select some points (only in 2D mode) and display a histogram or a distplot of the selected point.

![explore](https://github.com/diecalsa/Dash/blob/develop/src/explore_data2.gif)


#### Download the new dataset consisting of the original features and the new ones (principal components)

![Download](https://github.com/diecalsa/Dash/blob/develop/src/download_data.gif)




## Built with

* [Dash](https://dash.plotly.com/) -
* [Plotly](https://plotly.com/)
* [Scikit-Learn](https://scikit-learn.org/stable/)


## Authors

* **[Salim Chikh](https://www.linkedin.com/in/salim-chikh-48b679109/)** - 
* **[Diego Calvete](https://www.linkedin.com/in/diego-calvete-010532b5/)** -
