"""Lattice calculations, deployment scaling, and regression primitives."""

#reviewed version as of 20/09/2026, jai ramchandani.
#last edited 26/08/2026, jai ramchandani

from __future__ import annotations

# These imports provide tools used below: temporary changes, copying cached
# lattices, caching expensive calculations, file hashing, and environment
# variables.
from contextlib import contextmanager # use this for the python invocation with something(): do this. 
# we use it in the optimised builder of the ORM, and it calls with hoisted_tracking(): build ORM. It is 
# a temporary optimiser, uses caching. 
from copy import deepcopy # deepcopy makes an identical, completely independent copy of something - 
# we use this in ORM generation, and it prevents changes to ring A being applied to ring A' (deepcopy of A)
from functools import lru_cache # this caches input output pairs. E.g. if f(5)=8, and we re-call f(5), we get 8
# without any computation. Note for jai - investigate if there is a limit to this, if it is ORM's, we should not
#endlessly cache on the VM
import hashlib #sha256 hash, verifies that the regression files are indempotent to verified version
import os #useful for networking, classic systems stuff like setting threads etc. 
from pathlib import Path #helpful for file management. 
from .safety import validate_id_state #use this to make the ID state safe before using in calculations
import numpy as np #optimised math, normal np
import scipy.io as sio #matlab scipy, loads .mat gaptofield, m66 fits.

from . import config #imports config from same module. 
from .types import ORM #imports the shared ORM type from the same package


# Numerical kick policy and machine metadata have one configuration source.
KICK_RAD = config.KICK_RAD
_FIT_CACHE: dict[str, object] = {}

# checks the id state against a validated id states to get rid of any rubbish before it hits prod (
# too many id's, not enouggh, infinite values or nans)
def _state_array(state: dict[str, float] | np.ndarray) -> np.ndarray:
    """Turn either a dictionary or NumPy array into one validated array."""
    if isinstance(state, dict):
        missing = [name for name in config.ID_NAMES if name not in state] # collect each expected ID name absent from the state argument
        extra = sorted(set(state).difference(config.ID_NAMES)) # if there are incorrect, extra ID names in the state argument, add to extra array
        if missing or extra: # if missing or extra arrays are populated, raise error
            raise ValueError(f"ID state keys differ: missing={missing}, extra={extra}")
        out = np.array([float(state[name]) for name in config.ID_NAMES])
    else:
        out = np.asarray(state, float).ravel()
        if out.shape != (len(config.ID_NAMES),):
            raise ValueError(f"expected {len(config.ID_NAMES)} ID values, got shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("ID state contains non-finite values")
    return out

#uses the fit_cache from above, calls in m66 and ff, since that is all (I believe) we use this function 
# (fitcache) for in the whole twin.
# Review synthesis (Sam/Michael): this cache holds the loaded fit tables,
# not ID-dependent predictions. Each table is loaded independently so a
# partial load failure can be retried without accepting an incomplete cache.
def _fits() -> tuple[dict, dict]:
    """Load interpolation tables once and return them on later calls."""
    # Check the entries separately so that a partially populated cache can
    # recover. For example, if loading the feedforward file fails after the
    # M66 file has loaded, the next call retries only the missing FF entry
    # instead of treating the non-empty cache as complete and raising a
    # misleading KeyError.
    if "m66" not in _FIT_CACHE:
        _FIT_CACHE["m66"] = sio.loadmat(config.M66_FITS_MAT, squeeze_me=True, struct_as_record=False)
    if "ff" not in _FIT_CACHE:
        _FIT_CACHE["ff"] = sio.loadmat(config.QUAD_FF_MAT, squeeze_me=True, struct_as_record=False)
    return _FIT_CACHE["m66"], _FIT_CACHE["ff"]  # type: ignore[return-value]

# basically making np.interp more nice, we pass in a value we want to interpolate (x), 
# a set of x's, and a set of y's. np.interp does the clamping, all this does is sort for us. 
# jai note DONE 25/08/2026, patched for NaNs and for duplicate values in the x. 
def _interp_clamped(x: float, axis: np.ndarray, values: np.ndarray) -> float:
    """Interpolate safely, using the nearest endpoint outside the table."""
    # Convert the inputs to simple one-dimensional floating-point arrays.
    axis = np.asarray(axis, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if axis.size == 0 or axis.size != values.size:
        raise ValueError("interpolation axis and values must be non-empty and have equal length")
    if not np.isfinite(x) or not np.isfinite(axis).all() or not np.isfinite(values).all():
        raise ValueError("interpolation tables must contain only finite values")

    # np.interp expects the x-axis to be ordered, so sort the x-values and
    # apply exactly the same ordering to their matching y-values.
    order = np.argsort(axis)
    sorted_axis = axis[order]
    sorted_values = values[order]

    # Duplicate x-values are safe only when they describe the same y-value.
    # Identical duplicate rows are collapsed; conflicting rows are rejected.
    unique_axis, inverse, counts = np.unique(
        sorted_axis, return_inverse=True, return_counts=True
    )
    if np.any(counts > 1):
        unique_values = []
        for group in range(len(unique_axis)):
            group_values = sorted_values[inverse == group]
            if not np.all(group_values == group_values[0]):
                raise ValueError("duplicate interpolation x-values have conflicting y-values")
            unique_values.append(group_values[0])
        sorted_axis = unique_axis
        sorted_values = np.asarray(unique_values)

    return float(np.interp(x, sorted_axis, sorted_values))

# Modelled ID coverage is explicit in the model reference and matched to AO.

def m66_plane(
    state: dict[str, float] | np.ndarray,
    plane: str,
) -> np.ndarray:
    """Return table-derived M66 focusing for one plane.

    Keeping the plane selection in one helper lets the Stage 2h model
    substitute a smooth plane without maintaining separate Kx/Ky wrappers.
    """
    if plane not in {"Kx", "Ky"}:
        raise ValueError(f"plane must be 'Kx' or 'Ky', got {plane!r}")
    values = _state_array(validate_id_state(state))
    m66, _ = _fits()
    result = np.zeros(len(config.ID_NAMES))
    for j, name in enumerate(config.ID_NAMES):
        key = config.M66_KEY.get(name)
        if key is None:
            continue
        fit = m66[key]
        axis = np.ravel(fit.Gap).astype(float)
        result[j] = _interp_clamped(
            values[j], axis, np.ravel(getattr(fit, plane)).astype(float)
        )
    return result


def id_state_to_params(state: dict[str, float] | np.ndarray) -> dict[str, np.ndarray]:
    """Convert insertion-device gaps into model parameters."""
    values = _state_array(validate_id_state(state))
    _, ff = _fits()
    kx = m66_plane(values, "Kx")
    ky = m66_plane(values, "Ky")

    #init all FF effects to zero
    delta = {family: np.zeros(config.FAMILY_COUNTS[family[1:].upper()])
             for family in ("k" + name.lower() for name in config.QUAD_FAMILIES)}
    #if no FF effects, skip. three of the eight do not have FF fits at the moment. Note for jai (and toby)
    # look into re-fitting the FF effects, ask eugene. 
    for j, name in enumerate(config.ID_NAMES):
        key = config.QUAD_FF_KEY.get(name)
        if key is None:
            continue
        table = ff[key]
        axis = np.ravel(table.gaps).astype(float)
        #nested for loop, essentially here we have a 2d array which we are sum flattening into a 1d array. 
        # for each ID, for each quadrupole set (independent FF between each family), there is some correction.
        # sum net effects to get net FF effect at each quad.
        #note for jai, we could keep this on the hot path and run this optimally with either a smaller timestep
        # refresh rate, or smaller idFrob threshold. 
        for family, target in delta.items():
            rows = np.asarray(getattr(table, family), float)
            if rows.shape != (len(target), len(axis)):
                raise ValueError(f"feed-forward {key}.{family} shape differs from AO family count")
            for row in range(len(target)):
                target[row] += _interp_clamped(values[j], axis, rows[row])
                #sum the values as the net m66 effects from each ID, then  add the change to the quad 
                # strengths individually. note for jai, sums them linearly, thus assuming indep. between ID
                #. We have tested this effect on the ORM, but we (don't?) have a way of verifying quads. 
                #. look into this, should we add corrections/coupling? Stage 2h
    return {"Kx": kx, "Ky": ky, **{f"dK{name.lower()}": delta[f"k{name.lower()}"]
                                    for name in config.QUAD_FAMILIES}}

#essentially returns either binary or linear in expectation function which we can use to regress.
# thus, we do not have to use kernel ridge regression, because this theoretically does not have curves. 
def regression_features(state: dict[str, float] | np.ndarray) -> np.ndarray:
    """Create the numerical features used by the validated scan regression."""
    values = _state_array(validate_id_state(state))
    #jai - changed periods (undulator periods) to not be hardcoded, they are now in config.
    periods = config.GAP_FEATURE_PERIOD_MM
    result = np.zeros(len(config.ID_NAMES))
    for j, name in enumerate(config.ID_NAMES):
        if name in periods:
            #although this looks systematic, it is not a perfect electromagnetic calculation. We expect the 
            # gaps converted through the exponential gaps, and SCW B^2 (SCW mag field strength ^2), to have
            # a roughly linear impact on the orbit. 
            result[j] = np.exp(-2 * np.pi * values[j] / periods[name])
        elif name in config.FIELD_SQUARED_IDS:
            result[j] = values[j] ** 2
        else:
            raise RuntimeError(f"no regression feature transform configured for {name}")
    return result

# shows the locations around the ring for every element group (QFA, SFA, etc.)
def family_indices(ring) -> dict[str, np.ndarray]:
    """Index every element family in one traversal of the lattice."""
    grouped: dict[str, list[int]] = {}
    for index, element in enumerate(ring):
        grouped.setdefault(element.FamName, []).append(index)
    return {
        name: np.asarray(indices, dtype=np.uint32)
        for name, indices in grouped.items()
    }


def _family(indices: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Return one family, or an empty uint32 array when it is absent."""
    return indices.get(name, np.empty(0, dtype=np.uint32))


def _validate_family_indices(indices: dict[str, np.ndarray]) -> None:
    """Reject a lattice whose validated production-family counts changed."""
    wrong = {
        name: (len(_family(indices, name)), expected)
        for name, expected in config.EXPECTED_FAMILY_COUNTS.items()
        if len(_family(indices, name)) != expected
    }
    if wrong:
        detail = ", ".join(
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in wrong.items()
        )
        raise RuntimeError(f"unexpected lattice family counts: {detail}")

#loads in the ring, uses all of the configs from .config. Nothing hardcoded, 
def _load_ring(mode: int):
    """Load the accelerator lattice and apply the selected machine mode."""
    try:
        import at
    except ImportError as exc:
        raise RuntimeError("Accelerator Toolbox is required to calculate an AT ORM") from exc
    #using try except to give a more detailed error msg if no at pkg installed
    ring = at.Lattice.load(config.LATTICE_MAT, use="ring")
    ring.energy = config.ENERGY_EV
    settings = config.MODE_SETTINGS[mode]
    for element in ring:
        element.Energy = config.ENERGY_EV
        if element.FamName in config.QUAD_FAMILIES:
            strength = settings[element.FamName]
            element.PolynomB[1] = strength
            element.K = strength
        elif element.FamName in config.SEXT_FAMILIES:
            element.PolynomB[2] = settings[element.FamName]
    return ring
#build ring family by family

#caches the ring if it changes. Stores up to four, note for jai check size. Should not nearly be a problem
# further, pigeonhole we will never need four unless a new operating state is added (we only have 1,2, 10.)
@lru_cache(maxsize=4)
def _cached_ring(mode: int):
    """Keep up to four loaded machine-mode lattices in memory."""
    if mode not in config.MODE_SETTINGS:
        raise ValueError(f"machine mode must be one of {sorted(config.MODE_SETTINGS)}")
    return _load_ring(mode)


@lru_cache(maxsize=4)
def _cached_family_indices(mode: int) -> dict[str, np.ndarray]:
    """Build and validate one immutable family map per machine mode."""
    indices = family_indices(_cached_ring(mode))
    _validate_family_indices(indices)
    for positions in indices.values():
        positions.flags.writeable = False
    return indices

#loads one of the modes, caches it using the above function if it has not been cached as of yet. note for jai,
# check runtime - if the first computations spills over, we may benefit from precomputing on startup (not 
# from operating modes given the rarity of their changes, but another cached item, )

#optionally makes a deep copy, we use this in stages where the lattice is further modified and we do not want it 
# spilling back into the cached ring . 
def load_ring(mode: int = config.DEFAULT_MACHINE_MODE, *, copy: bool = True):
    """Return a lattice, optionally making a safe independent copy."""
    ring = _cached_ring(mode)
    return deepcopy(ring) if copy else ring


def apply_id_state(state: dict[str, float] | np.ndarray, *, mode: int = config.DEFAULT_MACHINE_MODE, feedforward: bool = True):
    """Apply insertion-device effects and optional quadrupole feed-forward."""
    #loads ring into at
    ring = load_ring(mode)
    indices = _cached_family_indices(mode)
    #changes the id state into the params (feed forward, m66 fits for each ID)
    params = id_state_to_params(state)
    #runs a check - is there 14 ID's in the ring? If no, stops and throws error. note for jai: quad count 
    # is from .config, why is this not? Should make this dynamic too
    markers = _family(indices, "ID")
    if markers.shape != (config.N_ID_MARKERS,):
        raise RuntimeError(
            f"expected {config.N_ID_MARKERS} ID markers, found {len(markers)}"
        )
    #for each ID, if an m66 fit does exist, construct an array with the kx and ky effects.
    for name, marker_index in config.ID_MARKER_INDEX.items():
        j = config.ID_NAMES.index(name)
        matrix = np.eye(6)
        matrix[1, 0] = params["Kx"][j]
        matrix[3, 2] = params["Ky"][j]
        ring[int(markers[marker_index])].M66 = matrix
    # for each quad, if feedforward exists, use the summed effects and apply them to each quad (by adding 
    # ff onto the base strength)
    if feedforward:
        for family in config.QUAD_FAMILIES:
            key = "dK" + family.lower()
            family_positions = _family(indices, family)
            if family_positions.shape != (config.FAMILY_COUNTS[family],):
                raise RuntimeError(f"expected {config.FAMILY_COUNTS[family]} {family} elements")
            for j, index in enumerate(family_positions):
                ring[int(index)].PolynomB[1] = config.MODE_SETTINGS[mode][family] + params[key][j]
    return ring


@contextmanager #contextmanager, lets us use the with something(): do XYZ. 
#(in this form, with _hoisted_tracking(): build_orm().  

#note for reviewers, this is the optimised part of the ORM build. stops a lot of recomputation. 
# higher level - it does this by using caching subfunctions. We then swap these into AT instead of their
# existing helpers, then run the orm generation and swap them back.
def _hoisted_tracking(ring):
    """Cache invariant pyAT lattice scans during one matrix build."""
    import at.tracking.track as tracking
    from at.tracking.utils import variable_refs #two functions give access to pyAT's internal
    #tracking functions, allow us to check if the matrix is variable and such. These functions
    # are lower level than something like findorbit6(), but it is used to save more duplicit computation.

    # If the lattice has variable elements, temporarily replacing these
    # tracking helpers would not be safe.
    if sum(variable_refs(ring)) != 0: #if the lattice is a variable lattice, 
    #skip everything below this until end of function, throws no errors, and uses the normal pyAT run path.
        yield
        return
    #everything below here until end of function is the true optimisation. We temporarily change the behaviour 
    #of pyAT by replacing two old functions with new ones, and then swapping the old ones back in to ensure
    # the base case behaviour of pyAT is unchanged. 
    original_disable = tracking.disable_varelem #stores the old pyAT function as original_disable
    original_index = tracking.get_uint32_index #stores the old pyAT function as original_index
    memo: dict[tuple, np.ndarray] = {} #temporary cache, which stores previously calculated 
    #key value pairs without computation. discarded after function invoked. We use within 
    # subfunctions of this function to save time on computing. 

    #original_disable is unneeded, we do not call it because at the start we checked if lattice is
    # variable, if it has made it here it is not. We thus temporarily make it an empty lambda

    #original index just gets the indices of the elements in the AT format, 32-bit ints.


    #subfunction, optimised version of original_index pyAT. 
    def cached_index(lattice, refpts, **kwargs): #inputs are the lattice, refpts (requested points such 
        #as BPM posiitons). We never specify this, but it is internally called while an ORM is being generated.
        
        #handles the possible types of the refpts.

        #next block is identical to original_index, but memorises the input output pairs. if an input 
        # has been cached, it is significantly quicker and we thus just lookup. If not in cache, we
        # recompute. 
        if isinstance(refpts, np.ndarray):
            key = (id(lattice), refpts.dtype.str, refpts.tobytes())
        elif isinstance(refpts, (list, tuple)):
            key = (id(lattice), tuple(refpts))
        else:
            return original_index(lattice, refpts, **kwargs)
        if key not in memo:
            memo[key] = original_index(lattice, refpts, **kwargs)
        return memo[key]

    tracking.disable_varelem = lambda lattice: lattice #typically, this function takes a while in pyAT. 
    # we have already checked the fact that there are no variable elements in the array - and we cannot
    # be bothered writing the internal function again with a monkeypatch. Thus, we just replace the slow
    # function (disable varelem) with do_nothing(). Returns the lattice. 
    tracking.get_uint32_index = cached_index #replaces the original pyAT function with the optimised one wrote above. 

    try:
        yield #using contextmanager, this pauses _hoisted_tracking and then lets
        #the code in the "with" block run (building the ring. )
    finally:
        tracking.disable_varelem = original_disable #after the ring is built with the optimised method, swap back 
        #to the old functions. 
        tracking.get_uint32_index = original_index

# thiis function just concats the hcms and vcms, often we need to call them together. checks the output
# for safety.
def _corrector_indices(indices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal and vertical corrector positions in the ring."""
    def positions(family):
        return np.sort(np.concatenate([_family(indices, name)
                                       for name in config.CORRECTOR_LATTICE_FAMILIES[family]]))
    hcm, vcm = positions("HCM"), positions("VCM")
    if hcm.shape != (config.N_HCM,) or vcm.shape != (config.N_VCM,):
        raise RuntimeError(f"corrector counts are {len(hcm)}/{len(vcm)}, expected {config.N_HCM}/{config.N_VCM}")
    return hcm, vcm


#finite difference calculation of each ORM column. Essentially, we are asking: if we step corrector i
# by kick size specified at top of file, how much does each BPM move (1 through 98). 

#when making an orm, we will call this in an optimised way with _hoisted_tracking, to add caching 
# and reduce recomputation.
# Review synthesis (Toby): central differences use both kick signs and
# restore the original corrector angle in finally, including orbit failures.
def _column(ring, index: int, plane: int, bpm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one ORM column using a small positive and negative kick."""
    import at #we import AT here to not force AT import unless a function running AT is ran. ~compute free.

    saved = np.array(ring[index].KickAngle, float)
    try:
        ring[index].KickAngle[plane] = saved[plane] + KICK_RAD # applies the positive kick, offset from whatever the corrector was already set to
        _, plus = at.find_orbit6(ring, refpts=bpm) # finds the 6D closed orbit. plus is an (n_bpm, 6) array of the orbit at each monitor
        ring[index].KickAngle[plane] = saved[plane] - KICK_RAD # applies the negative kick, offset from whatever the corrector was already set to
        _, minus = at.find_orbit6(ring, refpts=bpm) # finds the 6D closed orbit. minus is an (n_bpm, 6) array of the orbit at each monitor
    finally:
        ring[index].KickAngle[:] = saved # restores the original kick even if find_orbit6 fails to converge and raises error

    # Return the central finite-difference derivatives dx/dtheta and dy/dtheta in m/rad.
    return ((plus[:, 0] - minus[:, 0]) / (2 * KICK_RAD), (plus[:, 2] - minus[:, 2]) / (2 * KICK_RAD))

#multiprocessing, not multithreading - mp is creating more python processes and splitting work, 
# mt is splitting one python process and sharing memory. Should we use mt to bound memory?
# ask eugene about strict memory reqs, current answer is no because we dont want threads colliding
_WORKER: dict[str, object] = {}



def _worker_init(ring, bpm: np.ndarray) -> None:
    """Give each multiprocessing worker its lattice and BPM positions."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")#OMP and MKL, two inbuilt threading/multiprocessing
    # systems, we set to 1 because we are doing all of the multiprocessing determination ourselves.
    # we do not want nested multiprocessing, too much memory
    _WORKER["ring"] = ring #give the worker dictionary the ring and the bpm information. 
    _WORKER["bpm"] = bpm

#uses the worker dictionary to calculate column. We pass in a tuple "task", contaning the 
# corrector location and the kick plane. we get ring and bpm info from the worker dict. 
def _worker_column(task: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Worker-process wrapper for calculating one ORM column."""
    return _column(_WORKER["ring"], task[0], task[1], _WORKER["bpm"])  # type: ignore[arg-type]

#full ORM build, uses optimised functions.  
def build_orm(ring, *, workers: int = 1, indices: dict[str, np.ndarray] | None = None) -> ORM:
    """Build an ORM; choose worker count from available memory, not CPU count alone."""
    indices = family_indices(ring) if indices is None else indices
    bpm = _family(indices, "BPM") #gets the locations of all of the BPM's
    #checks that the bpm data is expected shape
    if bpm.shape != (config.N_BPM,):
        raise RuntimeError(f"expected {config.N_BPM} BPMs, found {len(bpm)}")
    #gets the location of the correctors (horiz and vert)
    hcm, vcm = _corrector_indices(indices)
    #makes each "task" array, contains the corrector location and kick plane (0 for HCM, 1 for VCM)
    #note for reviewer - investigate corrector angular shift in Stage 2h
    tasks = [(int(index), 0) for index in hcm] + [(int(index), 1) for index in vcm]
    #lazy import multiprocessing if we are using more than 1 worker.
    if workers > 1:
        import multiprocessing as mp

    #creates a pool of workers, gives them the ring and BPM data to store in worker dict. 
    #enables the caching and tracking optimisation.
        with _hoisted_tracking(ring), mp.get_context("fork").Pool(
            workers, initializer=_worker_init, initargs=(ring, bpm)
        ) as pool:
            #combine all of the worker results into an ORM.
            result = pool.map(_worker_column, tasks, chunksize=1)
    else:
        #calculate the ORM normally, Rxx, Ryx, Rxy, Ryy.
        with _hoisted_tracking(ring):
            result = [_column(ring, index, plane, bpm) for index, plane in tasks]
    split = len(hcm)
    #merge the quadrants into one ORM
    return ORM(
        np.column_stack([value[0] for value in result[:split]]),
        np.column_stack([value[1] for value in result[:split]]),
        np.column_stack([value[0] for value in result[split:]]),
        np.column_stack([value[1] for value in result[split:]]),
        "m/rad",
        #regardless of whether mp was used, checks whether ORM has no NaN's or inf and is right shape. 
    ).check()

#we do not cache ID aware ORM's, as the input is infinite param. No point saving a few ms of runtime if ORM is
# on an outdated ID state. 

@lru_cache(maxsize=3)
def _cached_static_at_orm(mode: int) -> ORM:
    """Build one immutable static ORM per machine mode."""
    result = build_orm(
        load_ring(mode), workers=1, indices=_cached_family_indices(mode)
    )
    for block in (result.Rxx, result.Ryx, result.Rxy, result.Ryy):
        block.flags.writeable = False
    return result


def static_at_orm(mode: int = config.DEFAULT_MACHINE_MODE, workers: int = 1) -> ORM:
    """Return the cached static ORM; ``workers`` is retained for compatibility."""
    del workers
    return _cached_static_at_orm(mode)

#function to build ORM with Id state in mind in AT. note for jai - make this built into build_orm (optional param?)
def id_aware_at_orm(state: dict[str, float] | np.ndarray, *, mode: int = config.DEFAULT_MACHINE_MODE, workers: int = 1) -> ORM:
    """Build an ORM after applying the current insertion-device state."""
    return build_orm(
        apply_id_state(state, mode=mode),
        workers=workers,
        indices=_cached_family_indices(mode),
    )

# Converts units with the fixed Rxx/Ryy fudge factors used by the OFB.
def at_to_ofb(orm: ORM) -> ORM:
    """Apply the facility Rxx/Ryy fudge factors; never use amp2k."""
    orm.check() #check validity, probably duplicate call because it is called outside the conversion note for jai
    #, but it is fine because very light computation
    if orm.units == "nm/A":#check if already in LOCO units, if so then skip all. 
        return orm
    if orm.units != "m/rad": #is this not checked in validity? .check() note for jai
        raise ValueError(f"cannot deploy ORM with units {orm.units!r}")
    return ORM(
        orm.Rxx * config.OFB_RXX_SCALE,
        orm.Ryx * config.OFB_RXX_SCALE,
        orm.Rxy * config.OFB_RYY_SCALE,
        orm.Ryy * config.OFB_RYY_SCALE,
        "nm/A",
    ).check()
#checks at the end after conversion. 

#averages the ORM's safely
def mean_orm(values: list[ORM]) -> ORM:
    """Average compatible ORMs after checking their units."""
    if not values: #checks for safety and emptiness
        raise ValueError("cannot average an empty ORM history")
    units = values[0].units
    for value in values:
        value.check()
        if value.units != units:
            raise ValueError("cannot average ORMs with different units")
        #if safe, mean for all values.
    return ORM(
        np.mean([value.Rxx for value in values], axis=0),
        np.mean([value.Ryx for value in values], axis=0),
        np.mean([value.Rxy for value in values], axis=0),
        np.mean([value.Ryy for value in values], axis=0),
        units,
    ).check()

#adds orm's safely after checking for units. Default scale 1 for RHS, thus pure addition. 
def add_orm(left: ORM, right: ORM, scale: float = 1.0) -> ORM:
    """Add two compatible ORMs, optionally scaling the second one."""
    if left.units != right.units:
        raise ValueError("cannot combine ORMs with different units")
    #adds and checks ORM validity after. 
    return ORM(left.Rxx + scale * right.Rxx, left.Ryx + scale * right.Ryx, left.Rxy + scale * right.Rxy, left.Ryy + scale * right.Ryy, left.units).check()

#calculate artifact hash. Used only for regression artifact
# the "regression artifact" is a big dictionary which stores information in stages 6 and 7 (currently),
# which is used in the ML. It is a result of the training on the 21 results. 
def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)#caches the result, there is only one regression artifact so we only need one cache.
def _regression_artifact() -> dict[str, np.ndarray]:
    """Load and verify the frozen regression model."""
    if not config.REGRESSION_ARTIFACT.exists(): #checks whether the artifact exists in config
        raise RuntimeError(f"missing frozen regression artifact {config.REGRESSION_ARTIFACT}")
    #checks whether the hashes match, if not - something has changed.
    if _artifact_hash(config.REGRESSION_ARTIFACT) != config.REGRESSION_ARTIFACT_SHA256:
        raise RuntimeError("frozen regression artifact SHA-256 mismatch")
    #loads in the file, "allow pickle" simply stops python loading in arbitrary elements. 
    with np.load(config.REGRESSION_ARTIFACT, allow_pickle=False) as data:
        #copies the artifact into a dict of numpy arrays (e.g. stage6_intercept = )
        artifact = {key: np.array(data[key]) for key in data.files}
    #checks that the artifact versions match up,meaning what we trained it on and what we are expecting. 
    if str(artifact["version"]) != "scan-ridge-v1":
        raise RuntimeError("unsupported regression artifact version")
    if str(artifact.get("feature_version", "")) != config.REGRESSION_FEATURE_VERSION:
        raise RuntimeError("unsupported regression feature version")
    return artifact


#uses the artifact to return an ORM (stage6 and stage7.)
# Review synthesis (Nick/Toby): these coefficients belong to the frozen
# artifact and its feature contract. Active service Stage 7a predictions use
# ActiveStage7 instead; changing a feature transform requires a new fit.
def regression_prediction(state: dict[str, float], prefix: str) -> ORM:
    # takes in an input for ID state, then prefix (stage6 or stage7)
    """Use the verified regression artifact to predict an ORM."""
    artifact = _regression_artifact() #loads in the artifact using the above function while verifying hash
    raw = _state_array(validate_id_state(state)) #validates ID state and renames it
    support_min = artifact["support_min"] #min value used during training
    support_max = artifact["support_max"] #max value used during training
    tolerance = artifact["support_tolerance"] #tolerance (how far can we extrapolate?)
    #The measurement procedure dictated that we went from min to max at all ID strengths, as of 
    # 10/08/2026, so without changes, we should not run into issues here. Include in re-calibration policy. 
    if np.any(raw < support_min - tolerance) or np.any(raw > support_max + tolerance):
        #if outside extrapolation bounds raise error
        details = [name for name, value, low, high, tol in zip(config.ID_NAMES, raw, support_min, support_max, tolerance) if value < low - tol or value > high + tol]
        raise ValueError(f"ID state is outside frozen scan support: {', '.join(details)}")
    #load in the features which we regress against
    features = regression_features(state)
    #normalises the features such that each of the ID's has same impact. 
    scaled = (features - artifact[f"{prefix}_feature_mean"]) / artifact[f"{prefix}_feature_scale"]
    #actually does the calculation (pred = intercept+features*coeffs)
    vector = artifact[f"{prefix}_intercept"] + scaled @ artifact[f"{prefix}_coef"]

    n_h = config.N_BPM * config.N_HCM # number of BPMs multiplied by number of HCM's (98*42 = 4116)
    rxx = vector[:n_h].reshape(config.N_BPM, config.N_HCM) #reshape the vector prediction into horiz RM
    ryy = vector[n_h:].reshape(config.N_BPM, config.N_VCM)#reshape the vector prediction into vert RM
    result = ORM(# as type ORM, return the full ORM
        rxx,
        np.zeros_like(rxx), #make ryx null
        np.zeros((config.N_BPM, config.N_VCM)), #make rxy null
        ryy,
        "nm/A",
    )
    return result.check()
