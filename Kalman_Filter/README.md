
# Kalman Filters

#### What are Kalman Filters?
There has been quite remarkable advancements happening around the world. Most of the advanced machines and
instruments are majorly dependent on the numerous sensors inorder to estimate the unknown variables
completely based on measurements. I would like to take an example of a GPS sensor, which provides
us the estimation for location and velocity, but they are the hidden(unknown) variables while the
differentiable time of the satellites' signals' arrival are the measurements.

#### How are they helping us?
In the presence of uncertainity, estimating these hidden variables and providing the precise and accurate estimate
is challenging where we are having external factor that indulge with our sensor data which includes
thermal noise, slight chnages in sensor positions, environmental factors etc. Kalman filters are 
considered to be the most important and commonly used estimation algorithms, which not only
produces estimates from the ocean of uncertain and inaccurate data, but also predict where the next 
pose will be.

The filter is named after Rudolf E. Kálmán (May 19, 1930 – July 2, 2016).
In 1960, Kálmán published his famous paper describing a recursive solution to the discrete-data linear filtering problem.


## Installation

I have Anaconda distribution installed in my system, which also includes the Jupyter notebook.
However, I am proceeding through these following steps:

#### Create a virtual environment

(Using Anaconda Prompt) To create a new environment by using the below provided command:

```bash
  conda create -n yourenvname python=3.6
```

For more information, you may check here:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

After creating the new environment, we'd have to activate to enter the newly created environment

```bash
  activate yourenvname
```

#### Libraries we'd have to install before proceeding:

Library Name  | Command
------------- | -------------
Numpy         | ```bash conda install numpy```
Matplotlib    | ```bash conda install matplotlib```
Ipython       | ```bash conda install Ipython```

Or you may also use Jupyter Notebook for the same.

After successfull installation of these libraries, we are good to proceed. 