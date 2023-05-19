import matplotlib.pyplot as plt
import numpy, math

G = 1

def PolyCoefficients(x, coeffs):
    # Código obtido da internet em https://stackoverflow.com/questions/37352098/plotting-a-polynomial-using-matplotlib-and-coeffiecients
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def addToLimitVector(list, x):
    # Adds to a vector keeping it to a maximum size
    vecLim = 10000000
    if (len(list) >= vecLim):
        del list[0]
    list.append(x)

fig, ax = plt.subplots()
circle = plt.Circle((0, 0), 1, color='orange')

def plot_single(x, y):
    plt.ion()
    ax.cla()
    ax.add_patch(circle)
    ax.set_title("Position in relation to planet 2")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(x, y, color = 'blue')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.pause(0.000000000000000000000000001)


def plot_both(x1, y1, x2, y2, n):
    fig.clf()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x1, y1, label='Planeta 1')
    plt.plot(x2, y2, label='Planeta 2')
    plt.legend()
    if n == 0:
        plt.title("Planets positions")
        plt.savefig("Planets movement")
    elif n == 1:
        plt.title("Planets positions on the center of mass")
        plt.savefig("Planets movement on the center of mass")
    plt.clf()

def internalProduct(a1_x, a1_y, z1_x, z1_y):
    
    return ((a1_x - z1_x)**2) + ((a1_y - z1_y)**2)

def f1_x(a2_x):
    return a2_x

def f1_y(a2_y):
    return a2_y

def f2_x(a1_x, a1_y, z1_x, z1_y):
    return -m2 * ((a1_x - z1_x) / (internalProduct(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def f2_y(a1_x, a1_y, z1_x, z1_y):
    return -m2 * ((a1_y - z1_y) / (internalProduct(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def g1_x(z2_x):
    return z2_x

def g1_y(z2_y):
    return z2_y

def g2_x(a1_x, a1_y, z1_x, z1_y):
    return -m1 * ((z1_x - a1_x) / (internalProduct(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def g2_y(a1_x, a1_y, z1_x, z1_y):
    return -m1 * ((z1_y - a1_y) / (internalProduct(a1_x, a1_y, z1_x, z1_y) ** (3/2)))


def euler_aperf(a1_x0, a1_y0, z1_x0, z1_y0, a2_x0, a2_y0, z2_x0, z2_y0):
    k = 0

    p1_x.append(a1_x0)
    p1_y.append(a1_y0)
    p2_x.append(z1_x0)
    p2_y.append(z1_y0)
    a1_xk = a1_x0
    a1_yk = a1_y0
    a2_xk = a2_x0
    a2_yk = a2_y0
    z1_xk = z1_x0
    z1_yk = z1_y0
    z2_xk = z2_x0
    z2_yk = z2_y0

    while k < limite:
        a1_x_step = ((h/2) * (f1_x(a2_xk) + f1_x(a2_xk + h * f2_x(a1_xk, a1_yk, z1_xk, z1_yk))))
        a1_y_step = ((h/2) * (f1_y(a2_yk) + f1_y(a2_yk + h * f2_y(a1_xk, a1_yk, z1_xk, z1_yk))))
        a2_x_step = ((h/2) * (f2_x(a1_xk, a1_yk, z1_xk, z1_yk) + f2_x(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        a2_y_step = ((h/2) * (f2_y(a1_xk, a1_yk, z1_xk, z1_yk) + f2_y(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        z1_x_step = ((h/2) * (g1_x(z2_xk) + g1_x(z2_xk + h * g2_x(a1_xk, a1_yk, z1_xk, z1_yk))))
        z1_y_step = ((h/2) * (g1_y(z2_yk) + g1_y(z2_yk + h * g2_y(a1_xk, a1_yk, z1_xk, z1_yk))))
        z2_x_step = ((h/2) * (g2_x(a1_xk, a1_yk, z1_xk, z1_yk) + g2_x(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        za_y_step = ((h/2) * (g2_y(a1_xk, a1_yk, z1_xk, z1_yk) + g2_y(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))

        a1_xk += a1_x_step
        a1_yk += a1_y_step
        a2_xk += a2_x_step
        a2_yk += a2_y_step
        z1_xk += z1_x_step
        z1_yk += z1_y_step
        z2_xk += z2_x_step
        z2_yk += za_y_step

        k += 1
        p1_x.append(a1_xk)
        p1_y.append(a1_yk)
        p2_x.append(z1_xk)
        p2_y.append(z1_yk)
        addToLimitVector(r_px, a1_xk - z1_xk)
        addToLimitVector(r_py, a1_yk - z1_yk)

        plot_single(r_px, r_py)
        

def centerOfMass():
    ''' SETTING THE CENTER OF MASS '''
    # CORREÇÃO PARA O CENTRO DE MASSA
    cm_x = []
    cm_y = []

    c_p1_x = []
    c_p1_y = []
    c_p2_x = []
    c_p2_y = []

    # Obtendo o centro de massa
    for i in range(len(p1_x)):
        cm_x.append(((p1_x[i] * m1) + (p2_x[i] * m2)) / (m1 + m2))
        cm_y.append(((p1_y[i] * m1) + (p2_y[i] * m2)) / (m1 + m2))

    for i in range(len(p1_x)):
        c_p1_x.append(p1_x[i] - cm_x[i])
        c_p1_y.append(p1_y[i] - cm_y[i])
        c_p2_x.append(p2_x[i] - cm_x[i])
        c_p2_y.append(p2_y[i] - cm_y[i])

    plot_both(c_p1_x, c_p1_y, c_p2_x, c_p2_y, 1)

def RelativeToPlanet():
    plt.title("Position in relation to planet 2")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(r_px, r_py, color = 'blue')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.savefig("Position in relation to planet 2")
    plt.clf()


m1 = 1
m2 = 60
h = 0.2
limite = 10 ** 4

p1_x = []
p1_y = []
p2_x = []
p2_y = []
r_px = []
r_py = []

# Initial conditions: (p1_x, p1_y, p2_x, p2_y, p1_vx, p1_vy, p2_vx, p2_vy)
p1_xi = 1 # 
p1_yi = 5 # Initial positions of planet 1
p2_xi = 0 #
p2_yi = 0 # Initial positions of planet 2
p1_vxi = 1 # 
p1_vyi = -2 # Initial velocities of planet 1
p2_vxi = -2 # 
p2_vyi = -2 # Initial velocities of planet 2


vrelativa = (math.sqrt((p1_vxi - p2_vxi)**2 + (p1_vyi - p2_vyi)**2))
#p1_xi = (2 * m2 / vrelativa**2) + 1
prelariva = (math.sqrt((p1_xi - p2_xi)**2 + (p1_yi - p2_yi)**2))

kinecticEnergy = (1/2 * m1 * vrelativa**2)
potencialEnergy = -(G * m1 * m2) / prelariva

print(kinecticEnergy + potencialEnergy)

euler_aperf(p1_xi, p1_yi, p1_vxi, p1_vyi, p2_xi, p2_yi, p2_vxi, p2_vyi)
plot_both(p1_x, p1_y, p2_x, p2_y, 0)

#centerOfMass()
RelativeToPlanet()


sep = 5 # Separation between the interpolation pois

''' DISTANCE INTERPOLATION '''
tempo = []
tempo.append(0)
tempo.append(len(p1_x) // 8 * h)
tempo.append((len(p1_x) - 1) * h)

distancias = []

# Taking samples of the distance
distancias.append( math.sqrt((p1_x[0] - p2_x[0]) ** 2 + (p1_y[0] - p2_y[0]) ** 2))
distancias.append( math.sqrt((p1_x[len(p1_x)//8] - p2_x[len(p2_x)//8])**2 + (p1_y[len(p1_y)//8] - p2_y[len(p2_y)//8])**2))
distancias.append( math.sqrt((p1_x[len(p1_x)-1] - p2_x[len(p2_x)-1])**2 + (p1_y[len(p1_y)-1] - p2_y[len(p2_y)-1])**2))


''' INTERPOLATION OF THE DISTANCE '''
tempo = []
tempo.append(0)
tempo.append(len(p1_x)//8 * h)
tempo.append((len(p1_x)-1) * h)

# Taking samples from the distances
distancias = []
distancias.append( (p1_x[0] - p2_x[0])**2 + (p1_y[0] - p2_y[0])**2)
distancias.append( (p1_x[len(p1_x)//8] - p2_x[len(p2_x)//8])**2 + (p1_y[len(p1_y)//8] - p2_y[len(p2_y)//8])**2)
distancias.append( (p1_x[-1] - p2_x[-1])**2 + (p1_y[-1] - p2_y[-1])**2)

# Interpolation
a = []
for i in range(len(tempo)):
    linha = []
    for j in range(len(tempo)):
        linha.append(tempo[i] ** j)
    a.append(linha)

a_array = numpy.asarray(a)
b_array = numpy.asarray(distancias)
x_array = numpy.linspace(tempo[0], tempo[len(tempo) - 1], 100)
sol = numpy.linalg.solve(a_array, b_array)

# Graphs
plt.plot(x_array, PolyCoefficients(x_array, sol), color='orange')
plt.ylabel('Distance')
plt.xlabel('Time')
plt.savefig("Distance x Time")