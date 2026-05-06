import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres du pendule élastique (valeurs données en Q4)

k  = 17.8   # raideur du ressort [N/m]
m  = 0.2    # masse [kg]
r0 = 0.44   # longueur à vide du ressort [m]
g  = 9.8    # accélération de la pesanteur [m/s²]

# Équation du mouvement en coordonnées cartésiennes
# F = -k(||r|| - r0) * r/||r|| + m*g  (ressort + poids)
# Axe z orienté vers le bas (sens de la gravité)

def equation_de_mouvement(x, y, z, xp, yp, zp):
    r = np.sqrt(x**2 + y**2 + z**2)        # longueur du ressort
    coeff = -(k/m) * (1 - r0/r)            # force de rappel / m, projetée via (x,y,z)/r
    xpp = coeff * x
    ypp = coeff * y
    zpp = coeff * z + g                     # gravité selon +z (vers le bas)
    return xpp, ypp, zpp

# Forme vectorielle P = [x, y, z, x', y', z'] → dP/dt

def equation_de_mouvement_vectoriel(P):
    x, y, z, xp, yp, zp = P
    xpp, ypp, zpp = equation_de_mouvement(x, y, z, xp, yp, zp)
    return np.array([xp, yp, zp, xpp, ypp, zpp])

# Un pas de Runge-Kutta d'ordre 4

def rk4(P, dt):
    k1 = equation_de_mouvement_vectoriel(P)
    k2 = equation_de_mouvement_vectoriel(P + 0.5*dt*k1)
    k3 = equation_de_mouvement_vectoriel(P + 0.5*dt*k2)
    k4 = equation_de_mouvement_vectoriel(P +     dt*k3)
    return P + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Intégration temporelle : itère RK4 de t=0 à t_max avec un pas dt

def integrer(P_initial, dt, t_max):
    n = int(t_max/dt)
    t = np.linspace(0., n*dt, n+1)
    P = np.zeros((n+1, 6))       # tableau des états [x, y, z, x', y', z'] à chaque instant
    P[0] = P_initial
    for i in range(n):
        P[i+1] = rk4(P[i], dt)
    return t, P

# Calcul des énergies cinétique, potentielle (gravité + ressort) et totale

def energies(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    xp, yp, zp = P[:, 3], P[:, 4], P[:, 5]
    r = np.sqrt(x**2 + y**2 + z**2)
    Ec = 0.5*m*(xp**2 + yp**2 + zp**2)                # énergie cinétique
    Ep = -m*g*z + 0.5*k*(r - r0)**2                    # énergie potentielle (gravité + ressort)
    return Ec, Ep, Ec + Ep

# Trace 4 graphiques : coordonnées(t), trajectoire 3D, projection horizontale, énergies

def tracer(t, P, titre, fname):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    Ec, Ep, Et = energies(P)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(titre, fontsize=13, fontweight='bold')

    # r(t) et z(t) sur deux axes y
    a = fig.add_subplot(2, 2, 1)
    a.plot(t, r, 'b', label='r(t)')
    a.set_xlabel('t [s]')
    a.set_ylabel('r [m]', color='b')
    a.tick_params('y', colors='b')
    a2 = a.twinx()
    a2.plot(t, x, 'r', label='x(t)', lw=0.7)
    a2.plot(t, y, 'g', label='y(t)', lw=0.7)
    a2.set_ylabel('x, y [m]')
    a2.legend(loc='upper right', fontsize=8)
    a.set_title('Coordonnées en fonction du temps')
    a.grid(alpha=0.3)

    # Trajectoire 3D
    a = fig.add_subplot(2, 2, 2, projection='3d')
    a.plot(x, y, z, 'g', lw=0.5)
    a.plot([0], [0], [0], 'ks', ms=7, label='attache', zorder=5)
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')
    a.set_zlabel('z [m]')
    a.invert_zaxis()                    # z positif = bas, affiché vers le bas
    a.set_title('Trajectoire 3D')
    a.legend()

    # Projection dans le plan horizontal (x, y)
    a = fig.add_subplot(2, 2, 3)
    a.plot(x, y, 'm', lw=0.6)
    a.plot(0, 0, 'ks', ms=7, label='attache', zorder=5)
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')
    a.set_title('Projection horizontale (x, y)')
    a.set_aspect('equal')
    a.grid(alpha=0.3)
    a.legend()

    # Énergies
    a = fig.add_subplot(2, 2, 4)
    a.plot(t, Ec, 'b', label='E. cinétique')
    a.plot(t, Ep, 'r', label='E. potentielle')
    a.plot(t, Et, 'k', label='E. totale', lw=1.5)
    a.set_xlabel('t [s]')
    a.set_ylabel('Énergie [J]')
    a.set_title('Énergies')
    a.legend(loc='best')
    a.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname, dpi=130)
    plt.close(fig)

# --- Q3 : Vérifications par cas limites ---
# P_initial = [x, y, z, x', y', z']
# Axe z vers le bas, attache à l'origine

# Cas 1 : oscillation purement verticale (x=y=0, masse sous l'attache)
P_initial = np.array([0.0, 0.0, 0.66, 0.0, 0.0, 0.0])
t, P = integrer(P_initial, dt=0.001, t_max=10.0)
tracer(t, P, "Vérification 3D : oscillation verticale", "fig_3d/fig_3d_vertical.png")

# Cas 2 : k très grand → pendule simple dans le plan (x,z)
k = 100000.0
r_eq_grand = r0 + m*g/k
P_initial = np.array([r_eq_grand*np.sin(0.3), 0.0, r_eq_grand*np.cos(0.3), 0.0, 0.0, 0.0])
t, P = integrer(P_initial, dt=0.0005, t_max=10.0)
tracer(t, P, "Vérification 3D : k grand -> pendule simple", "fig_3d/fig_3d_pendule.png")
k = 17.8

# Cas 3 : pendule conique (rotation uniforme autour de l'axe z)
# Équilibre conique : r_c = r0 + mg/(k·cosθ), φ̇ = √(g/(r_c·cosθ))
theta_0 = 0.3
r_conique = r0 + m*g/(k*np.cos(theta_0))
phipoint_0 = np.sqrt(g/(r_conique*np.cos(theta_0)))
# Position initiale : masse dans le plan (x,z), vitesse selon y
x0 = r_conique*np.sin(theta_0)
z0 = r_conique*np.cos(theta_0)
vy0 = x0 * phipoint_0                  # v_tangentielle = R_horiz * φ̇
P_initial = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
t, P = integrer(P_initial, dt=0.001, t_max=10.0)
tracer(t, P, "Vérification 3D : pendule conique", "fig_3d/fig_3d_conique.png")

# --- Q4 : Résonance avec perturbation 3D ---

# Même CI que le 2D (r=0.66, θ=0.03) + vitesse azimutale pour sortir du plan
r_init, theta_init = 0.66, 0.03
x0 = r_init*np.sin(theta_init)
z0 = r_init*np.cos(theta_init)
vy0 = 0.5 * x0                         # petite vitesse hors-plan
P_initial = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
t, P = integrer(P_initial, dt=0.001, t_max=60.0)
tracer(t, P, "Résonance 3D  (k=17,8 ; m=0,2 ; r0=0,44)", "fig_3d/fig_3d_resonance.png")
