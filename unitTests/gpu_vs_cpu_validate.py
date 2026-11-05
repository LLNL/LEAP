"""GPU correctness checks for the LEAP ROCm port.

The CPU projectors cover only some geometry/model pairs (no cone-parallel, and
VD falls back to SF), so they are not used as the reference here. The port is
validated with three GPU checks that are sensitive to the texture-sampling
emulation:

  1. Finite: forward project, backproject and FBP are NaN/Inf-free across all 6
     geometries x {VD, SF}. The detector-edge 0/0 weights that CUDA hardware
     texturing tolerated must stay finite through the software-interp helper.
  2. Quantitative FBP: a uniform cylinder forward-projected then FBP-reconstructed
     must recover an interior value of 1.0. This is the decisive numeric gate; it
     exercises the volume/projection textures, the ramp filter and hipFFT, and the
     backprojection, end to end, and pins the absolute scale -- a wrong trilinear
     blend or a wrong hipFFT layout shifts the recovered value off 1.0.
  3. Adjointness: <A f, g> == <f, A^T g>; a wrong blend breaks it.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from leapctype import tomographicModels

numCols = 128
numRows = int(numCols * 25.0 / 24.0)
numAngles = 360
pixelSize = 1.03 * 2.0 * 250.0 / 512
cR = 0.5 * (numRows - 1)
cC = 0.5 * (numCols - 1)
sdd = 765.0
sod = 0.5 * sdd
tau = 0.0
tiltAngle = 1.0
pitch = 0.0

geom_names = ["parallel", "fan", "coneparallel", "cone-flat", "cone-curved", "modular"]
methods = ["VD", "SF"]

ADJ_TOL = 5e-3
FBP_TOL = 0.03   # cylinder interior recon within 3% of 1.0


def arr(a):
    return np.asarray(a, dtype=np.float32)


def set_geometry(lc, igeom):
    lc.reset()
    ang = lc.setAngleArray(numAngles, 360.0)
    if igeom == 0:
        lc.set_parallelbeam(numAngles, numRows, numCols, sod / sdd * pixelSize, sod / sdd * pixelSize, cR, cC, ang)
    elif igeom == 1:
        lc.set_fanbeam(numAngles, numRows, numCols, sod / sdd * pixelSize, pixelSize, cR, cC, ang, sod, sdd, tau)
    elif igeom == 2:
        lc.set_coneparallel(numAngles, numRows, numCols, pixelSize, sod / sdd * pixelSize, cR, cC, ang, sod, sdd, tau, pitch)
    elif igeom == 3:
        lc.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, cR, cC, ang, sod, sdd, tau, pitch, tiltAngle)
    elif igeom == 4:
        lc.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, cR, cC, ang, sod, sdd, tau, pitch)
        lc.set_curvedDetector()
    elif igeom == 5:
        lc.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, cR, cC, ang, sod, sdd, tau, 0.0, tiltAngle)
        lc.convert_to_modularbeam()


def finite(a):
    return bool(np.isfinite(arr(a)).all())


rng = np.random.default_rng(0)
results = []
fail = False

for igeom in range(6):
    for method in methods:
        lc = tomographicModels()
        set_geometry(lc, igeom)
        lc.set_default_volume(1.0)
        lc.set_gpu(0)
        lc.set_projector(method)

        # uniform cylinder phantom -> forward -> FBP must recover interior 1.0
        f = arr(lc.allocate_volume())
        nz, ny, nx = f.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        rad = min(ny, nx) * 0.3
        disk = ((yy - ny / 2.0) ** 2 + (xx - nx / 2.0) ** 2) < rad * rad
        f[:] = 0.0
        f[:, disk] = 1.0

        g = arr(lc.allocate_projections())
        lc.project(g, f)
        fbp = arr(lc.allocate_volume())
        lc.FBP(g, fbp)
        bp = arr(lc.allocate_volume())
        lc.backproject(g, bp)

        ok_finite = finite(g) and finite(fbp) and finite(bp)

        interior = ((yy - ny / 2.0) ** 2 + (xx - nx / 2.0) ** 2) < (0.6 * rad) ** 2
        recon_val = float(arr(fbp)[nz // 2][interior].mean())
        fbp_err = abs(recon_val - 1.0)

        # adjointness
        fr = arr(lc.allocate_volume())
        fr[:] = rng.random(fr.shape).astype(np.float32)
        gr = arr(lc.allocate_projections())
        gr[:] = rng.random(gr.shape).astype(np.float32)
        Afr = arr(lc.allocate_projections())
        lc.project(Afr, fr)
        Atgr = arr(lc.allocate_volume())
        lc.backproject(gr, Atgr)
        lhs = float(np.sum(arr(Afr) * gr))
        rhs = float(np.sum(fr * arr(Atgr)))
        adj = abs(lhs - rhs) / (abs(rhs) or 1.0)

        ok = ok_finite and (fbp_err < FBP_TOL) and (adj < ADJ_TOL)
        fail = fail or not ok
        results.append((geom_names[igeom], method, ok))
        print(f"{geom_names[igeom]:13s} {method}  finite={ok_finite}  fbp_interior={recon_val:.4f}"
              f" (err {fbp_err:.3f})  adjoint={adj:.2e}  {'PASS' if ok else 'FAIL'}", flush=True)

print("\n==== SUMMARY ====")
for name, method, ok in results:
    print(f"{name:13s} {method}  {'PASS' if ok else 'FAIL'}")
print("OVERALL:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
