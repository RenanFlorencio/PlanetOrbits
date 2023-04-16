# PlanetOrbits
Project to implement a method for solving the second order differential equations associated with a system of two planets.

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
