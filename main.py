import numpy as np
import matplotlib.pyplot as plt

# Paramètres du pendule élastique (valeurs données en Q4)

k  = 17.8   # raideur du ressort [N/m]
m  = 0.2    # masse [kg]
r0 = 0.44   # longueur à vide du ressort [m]
g  = 9.8    # accélération de la pesanteur [m/s²]

# Q2 : Résolution Numérique

# Équations du mouvement : renvoie les accélérations (r'', θ'')

def equation_de_mouvement(r, theta, rp, thetap):
    rpp = r*thetap**2 + g*np.cos(theta) - (k/m)*(r - r0)        # eq. radiale
    thetapp = -2*rp*thetap/r - g*np.sin(theta)/r             # eq. angulaire
    return rpp, thetapp

# Forme vectorielle P = [r, θ, r', θ'] devient dP/dt = [r', θ', r'', θ'']

def equation_de_mouvement_vectoriel(P):
    r, theta, rp, thetap = P
    rpp, thetapp = equation_de_mouvement(r, theta, rp, thetap)
    return np.array([rp, thetap, rpp, thetapp])

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
    P = np.zeros((n+1, 4))       # tableau des états [r, θ, r', θ'] à chaque instant
    P[0] = P_initial
    for i in range(n):
        P[i+1] = rk4(P[i], dt)
    return t, P

# Calcul des énergies cinétique, potentielle (gravité + ressort) et totale

def energies(P):
    r, theta = P[:, 0], P[:, 1]
    rp, thetap = P[:, 2], P[:, 3]
    Ec = 0.5*m*(rp**2 + (r*thetap)**2)         # énergie cinétique
    Ep = -m*g*r*np.cos(theta) + 0.5*k*(r - r0)**2      # énergie potentielle
    return Ec, Ep, Ec + Ep

# Trace 4 graphiques : coordonnées(t), trajectoire (x,y), portrait r(θ), énergies

def tracer(t, P, titre, fname):
    r, theta = P[:, 0], P[:, 1]
    x =  r*np.sin(theta)               # conversion polaire vers cartésien
    y = -r*np.cos(theta)
    Ec, Ep, Et = energies(P)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(titre, fontsize=13, fontweight='bold')

    # r(t) et θ(t) sur deux axes y
    a = ax[0, 0]
    a.plot(t, r, 'b', label='r(t)')
    a.set_xlabel('t  [s]')
    a.set_ylabel('r  [m]', color='b')
    a.tick_params('y', colors='b')
    a2 = a.twinx()
    a2.plot(t, theta, 'r', label=r'$\theta$(t)')
    a2.set_ylabel(r'$\theta$  [rad]', color='r')
    a2.tick_params('y', colors='r')
    a.set_title('Coordonnées en fonction du temps')
    a.grid(alpha=0.3)
 
    # Trajectoire (x,y)
    a = ax[0, 1]
    a.plot(x, y, 'g', lw=0.8)
    a.plot(0, 0, 'ks', ms=7, label='attache', zorder=5)
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')
    a.set_title('Trajectoire dans le plan')
    a.set_aspect('equal')
    a.grid(alpha=0.3)
    a.legend()
 
    # Portrait r(theta)
    a = ax[1, 0]
    a.plot(theta, r, 'm', lw=0.6)
    a.set_xlabel(r'$\theta$ [rad]')
    a.set_ylabel('r [m]')
    a.set_title(r'Portrait $r(\theta)$')
    a.grid(alpha=0.3)
 
    # Énergies
    a = ax[1, 1]
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

# Q3 : Vérifications par cas limites

# Cas 1 : oscillation purement verticale (θzero = 0, vitesses nulles)
P_initial = np.array([0.66, 0.0, 0.0, 0.0])
t, P = integrer(P_initial, dt=0.001, t_max=10.0)
tracer(t, P, r"Vérification : oscillation verticale ($\theta_0=0$, vitesses nulles)", "fig_2d/fig_vertical.png")

# Cas 2 : k très grand, on retrouve un pendule simple
k = 100000.0
r_eq_grand = r0 + m*g/k               # longueur d'équilibre equiv à r0 quand k très grand
P_initial = np.array([r_eq_grand, 0.3, 0.0, 0.0])
t, P = integrer(P_initial, dt=0.0005, t_max=10.0)
tracer(t, P, "Vérification : k grand -> pendule simple", "fig_2d/fig_pendule.png")
k = 17.8                              # restauration de k pour la suite

# Q4 : résonnance

P_initial = np.array([0.66, 0.03, 0.0, 0.0])
t, P = integrer(P_initial, dt=0.001, t_max=60.0)
tracer(t, P, "Résonance pendule élastique  (k=17,8 ; m=0,2 ; r0=0,44)", "fig_2d/fig_resonance.png")
