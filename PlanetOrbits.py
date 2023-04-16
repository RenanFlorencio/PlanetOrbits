import matplotlib.pyplot as plt
import numpy

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

def prod_int(a1_x, a1_y, z1_x, z1_y):
    
    return ((a1_x - z1_x)**2) + ((a1_y - z1_y)**2)

def f1_x(a2_x):
    return a2_x

def f1_y(a2_y):
    return a2_y

def f2_x(a1_x, a1_y, z1_x, z1_y):
    return -m2 * ((a1_x - z1_x) / (prod_int(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def f2_y(a1_x, a1_y, z1_x, z1_y):
    return -m2 * ((a1_y - z1_y) / (prod_int(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def g1_x(z2_x):
    return z2_x

def g1_y(z2_y):
    return z2_y

def g2_x(a1_x, a1_y, z1_x, z1_y):
    return -m1 * ((z1_x - a1_x) / (prod_int(a1_x, a1_y, z1_x, z1_y) ** (3/2)))

def g2_y(a1_x, a1_y, z1_x, z1_y):
    return -m1 * ((z1_y - a1_y) / (prod_int(a1_x, a1_y, z1_x, z1_y) ** (3/2)))


def euler_aperf(a1_x0, a1_y0, z1_x0, z1_y0, a2_x0, a2_y0, z2_x0, z2_y0):
    k = 0

    a1_xk = a1_x0
    a1_yk = a1_y0
    a2_xk = a2_x0
    a2_yk = a2_y0
    z1_xk = z1_x0
    z1_yk = z1_y0
    z2_xk = z2_x0
    z2_yk = z2_y0

    while k < limite:
        a1_xk1 = a1_xk + ((h/2) * (f1_x(a2_xk) + f1_x(a2_xk + h * f2_x(a1_xk, a1_yk, z1_xk, z1_yk))))
        a1_yk1 = a1_yk + ((h/2) * (f1_y(a2_yk) + f1_y(a2_yk + h * f2_y(a1_xk, a1_yk, z1_xk, z1_yk))))
        a2_xk1 = a2_xk + ((h/2) * (f2_x(a1_xk, a1_yk, z1_xk, z1_yk) + f2_x(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        a2_yk1 = a2_yk + ((h/2) * (f2_y(a1_xk, a1_yk, z1_xk, z1_yk) + f2_y(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        z1_xk1 = z1_xk + ((h/2) * (g1_x(z2_xk) + g1_x(z2_xk + h * g2_x(a1_xk, a1_yk, z1_xk, z1_yk))))
        z1_yk1 = z1_yk + ((h/2) * (g1_y(z2_yk) + g1_y(z2_yk + h * g2_y(a1_xk, a1_yk, z1_xk, z1_yk))))
        z2_xk1 = z2_xk + ((h/2) * (g2_x(a1_xk, a1_yk, z1_xk, z1_yk) + g2_x(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))
        z2_yk1 = z2_yk + ((h/2) * (g2_y(a1_xk, a1_yk, z1_xk, z1_yk) + g2_y(a1_xk + h * f1_x(a2_xk), a1_yk + h * f1_y(a2_yk), z1_xk + h * g1_x(z2_xk), z1_yk + h * g1_y(z2_yk))))

        a1_xk = a1_xk1
        a1_yk = a1_yk1
        a2_xk = a2_xk1
        a2_yk = a2_yk1
        z1_xk = z1_xk1
        z1_yk = z1_yk1
        z2_xk = z2_xk1
        z2_yk = z2_yk1

        #print(round(k*h, 2), end='  ')

        k += 1
        p1_x.append(a1_xk)
        p1_y.append(a1_yk)
        p2_x.append(z1_xk)
        p2_y.append(z1_yk)


def plota_linha_2(x1, y1, x2, y2, n):
    plt.xlabel('X')
    plt.ylabel('Y')
    if n == 0:
        plt.title('Posição dos planetas')
    else:
        plt.title('Posição dos planetas ajustada para o centro de massa')
    plt.plot(x1, y1, label='Planeta 1')
    plt.plot(x2, y2, label='Planeta 2')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.legend()
    plt.show()

m1 = 5
m2 = 8
h = 0.2
limite = 10 ** 5

p1_x = []
p1_y = []
p2_x = []
p2_y = []

# Inicial conditions: (p1_x, p1_y, p2_x, p2_y, p1_vx, p1_vy, p2_vx, p2_vy)
euler_aperf(2, 4, -5, -4, -1, 0, 1, 0)
plota_linha_2(p1_x, p1_y, p2_x, p2_y, 0)

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

plota_linha_2(c_p1_x, c_p1_y, c_p2_x, c_p2_y, 1)

sep = 5 # Separação entre os pontos da interpolação

''' DISTANCE INTERPOLATION '''
tempo = []
tempo.append(0)
tempo.append(len(p1_x) // 8 * h)
tempo.append((len(p1_x) - 1) * h)

distancias = []
distancias.append( (p1_x[0] - p2_x[0])**2 + (p1_y[0] - p2_y[0])**2)
distancias.append( (p1_x[len(p1_x)//8] - p2_x[len(p2_x)//8])**2 + (p1_y[len(p1_y)//8] - p2_y[len(p2_y)//8])**2)
distancias.append( (p1_x[len(p1_x)-1] - p2_x[len(p2_x)-1])**2 + (p1_y[len(p1_y)-1] - p2_y[len(p2_y)-1])**2)

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
plt.xlabel('Tempo')
plt.show()