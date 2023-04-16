# PlanetOrbits
Project to implement a method for solving the second order differential equations associated with a system of two planets in 2D.

<p align="center">
  <img src="https://user-images.githubusercontent.com/122649765/232264161-d5fdde47-cae5-40e3-8fc9-9d0eb16dffdc.png" width = "600" height = "450"/>
</p>

## Model

The movement of the planets in this model is defined by the following differential equation:

```math
m_1 (\textbf{P}''_1) = F_{12}
```
```math
m_2 (\textbf{P}''_2) = F_{12}
```

The force is as follow:

```math
F_{ij} = \frac{m_i m_j}{r^2} \hat{r}
```

Where $r$ is the distance between the two planets. Substituting the equations:

```math
m_1 (\textbf{P}''_1) = \frac{m_1 m_2}{|P_2 - P_1|^2} * \frac{P_2 P_1}{|P_2 - P_1|}
```
```math
m_2 (\textbf{P}''_2) = \frac{m_1 m_2}{|P_1 - P_2|^2} * \frac{P_1 P_2}{|P_1 - P_2|}
```

To apply the method, it is necessary to rewrite the equations:
```math
a'_1 = a_2
```
```math
z'_1 = z_2
```
Here $a_1$ is the position of planet 1 and $a_2$ is the velocity of planet 1. The same goes for z. Then:
```math
m_1 a'_2 = \frac{m_1 m_2 * (a_1 - z_1)}{ < a_1 - z_1; a_1 - z_1 > ^ {3/2}} 
```
```math
m_1 z'_2 = \frac{m_1 m_2 * (a_1 - z_1)}{ < a_1 - z_1; a_1 - z_1 > ^ {3/2}} 
```

This gives 4 equations. Splitting them into the $x$ and $y$ axis, there is a total of 8 equations that can be solved providing the initial conditions: position of velocity of both planets on the $x$ and $y$ axis.

From that it's possible to plot the position of the planets as a function of time as shown at the start of the description.

# Center of Mass

After having the vector of positions, it is possible to get the center of mass of each position and them subtract their value from the original position. 

From this, position of the planets relative to the center of mass is obteined:

<p align="center">
  <img src="https://user-images.githubusercontent.com/122649765/232264258-4bf1f57b-296f-4645-8eba-26b88f335e7c.png" width = "600" height = "450"/>
</p>

# Distance in function of time

Having the position at each time, it is also possible to obtain a plot of distance in function of time. For that, instead of calculating the distance for every
single point, I am taking the interpolation of three points (star, middle and end) and plotting them:

<p align="center">
  <img src="https://user-images.githubusercontent.com/122649765/232264415-228aa3a4-5f9a-42e8-855d-2162e26934d5.png" width = "600" height = "450"/>
</p>



