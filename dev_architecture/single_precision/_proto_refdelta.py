"""Scratch: validate the single-level reference+delta increment (companion note
sec.2-3) numerically before porting to a metamath graph. Confirms
  refdelta(Phi; Phi_ref) == woodbury(Phi)   exactly in f64, any Phi_ref,
and shows the f32 win: the increment avoids the ytNmy - FtNmy.mu cancellation.
Single pulsar, diagonal GP prior Phi (power-law red noise shape).
"""
import numpy as np

rng = np.random.default_rng(0)
n, m = 800, 20                       # n_toa, n_gp modes
F = rng.standard_normal((n, m))
Ninv = 1.0 / rng.uniform(0.5, 2.0, n)    # diagonal white-noise inverse
# big residual so |logL| ~ 1e5-1e6, exercising the cancellation
y = 30.0 * rng.standard_normal(n)

# fixed pieces (fold to f64 constants)
v = F.T @ (Ninv * y)                 # FtNmy
G = F.T @ (Ninv[:, None] * F)        # FtNmF
ytNmy = y @ (Ninv * y)
lN = -np.sum(np.log(Ninv))           # logdet N


def woodbury(Phi, dt=np.float64):
    Phi = Phi.astype(dt); G_ = G.astype(dt); v_ = v.astype(dt)
    C = np.diag(1.0 / Phi) + G_
    cf = np.linalg.cholesky(C)
    mu = np.linalg.solve(C, v_)                         # C^-1 v
    lS = 2 * np.sum(np.log(np.diag(cf)))
    lP = np.sum(np.log(Phi))
    quad = float(ytNmy) - float(v_ @ mu)
    return -0.5 * quad - 0.5 * (float(lN) + lP + lS)


def _u_w(Phi, dt):
    Phi = Phi.astype(dt); G_ = G.astype(dt); v_ = v.astype(dt)
    C = np.diag(1.0 / Phi) + G_
    u = np.linalg.solve(C, v_)
    w = (1.0 / Phi) * u                 # Phi^-1 u
    return u, w


def refdelta(Phi, Phi_ref, dt=np.float64):
    # reference (f64 constants)
    lnL_ref = woodbury(Phi_ref, np.float64)
    u_ref, w_ref = _u_w(Phi_ref, np.float64)
    S0 = np.eye(m) + Phi_ref[:, None] * G        # I + Phi_ref G  (rows scaled)
    # per-call (dt)
    Phi = Phi.astype(dt)
    u, w = _u_w(Phi, dt)
    dPhi = (Phi - Phi_ref.astype(dt))            # covariance-space increment
    dQ = float(np.sum(dPhi.astype(np.float64) * w.astype(np.float64) * w_ref))   # f64 accumulate
    X = np.linalg.solve(S0.astype(dt), (dPhi[:, None] * G.astype(dt)))
    sign, dLdet = np.linalg.slogdet(np.eye(m, dtype=dt) + X)
    dlnL = 0.5 * dQ - 0.5 * float(dLdet)
    return lnL_ref + dlnL


# --- f64 equivalence: refdelta == woodbury for any Phi, any Phi_ref ---
Phi_ref = 10 ** rng.uniform(-7, -5, m)
print("f64 equivalence (refdelta - woodbury), several Phi:")
for _ in range(5):
    Phi = 10 ** rng.uniform(-7, -5, m)
    a, b = refdelta(Phi, Phi_ref), woodbury(Phi)
    print(f"   logL={b:14.6f}   diff={a-b:+.3e}")

# --- f32 win: error vs f64 woodbury, refdelta-f32 vs woodbury-f32 ---
print("\nf32 abs error vs f64 woodbury truth:")
for _ in range(5):
    Phi = 10 ** rng.uniform(-7, -5, m)
    truth = woodbury(Phi, np.float64)
    wb32 = woodbury(Phi, np.float32)
    rd32 = refdelta(Phi, Phi_ref, np.float32)
    print(f"   |logL|={abs(truth):11.2f}   woodbury_f32={abs(wb32-truth):.3e}   "
          f"refdelta_f32={abs(rd32-truth):.3e}")
