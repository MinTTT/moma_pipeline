"""
Cell morphology from mask using distance-transform medial axis.

Ported from Moma_cellPose/cell_morphology_voronoi.py into the moma_pipeline
package (morphology/voronoi_dt.py); the local moma_dt_kernels import now uses
moma_pipeline.morphology.kernels. Behavior and public API are unchanged.

Algorithm:
  1. Zero-boundary smoothing, distance transform and skeletonization
  2. Main endpoint geodesic through branched skeletons
  3. Translation-invariant spine smoothing and equal-step interpolation
  4. Contour-constrained ribs, integrated pole refinement and mesh validation
  5. Estimated cell length = body spine length + both fitted pole radii

The legacy default retains its historical contour reconstruction. channel_axis
selects only the initial route, then runs contour-rib midpoint refinement (up to
15 iterations), an immutable initial contour reference, and terminal-only rib
removal. All poles use bounded inward fitting against frozen local contour arcs;
branches retain separate local radii. Failed final candidates restore the
initial reference and its original full-span endpoint neighborhoods. Measurements
are extracted from the final mesh; both historical pole_refinement choices
execute this mandatory channel pipeline.
"""

import numpy as np
from numba import njit
from moma_pipeline.morphology.kernels import terminal_body_indices
from scipy.interpolate import splrep, splev
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import find_contours
from skimage.morphology import skeletonize

# Accepted production implementation and shared reconstruction defaults.
DT_MORPHOLOGY_VERSION = "dt_spine_2026_10_28"
DEFAULT_STEP_SIZE = None
DEFAULT_SMOOTH_TOL = 0.05
CHANNEL_AXIS_TRIAL_VERSION = "channel_axis_2026_10_28_trial1"
# Light centerline smoothing for the channel-axis body (graph vertices).
CHANNEL_AXIS_SMOOTH_WINDOW = 5

# =============================================================================
# 1. Distance-Transform Medial Axis
# =============================================================================

CHANNEL_POLE_REFINEMENT_VERSION = "dt_channel_reference_bounded_poles_2026_11_06"
CHANNEL_POLE_SUPPORT_TOLERANCE = 1.0
MAX_BODY_ITERATIONS = 15
BODY_CONVERGENCE_TOL_PX = 0.25
BODY_UPDATE_FRACTION = 0.45


def dt_medial_axis(cell_mask, step_size=DEFAULT_STEP_SIZE,
                   smooth_tol=DEFAULT_SMOOTH_TOL,
                   return_geometry=False, spine_mode="default",
                   channel_axis=(0.0, 1.0), pole_refinement="none",
                   return_diagnostics=False):
    """Select channel route, refine contour body, fit poles, trim, then validate.

    Save a validated initial contour reference, preserve all seed endpoints,
    and fit every pole inward against its fixed local contour arc. Branches
    use independent frozen radii. Final-stage failure restores the exact initial
    reference. Legacy spine_mode=default retains its previous algorithm.
    """
    if pole_refinement not in {"none", "contour"}:
        raise ValueError("pole_refinement must be none or contour")
    if spine_mode == "channel_axis":
        mesh,radii,mode,diagnostic=_channel_contour_pipeline(
            np.asarray(cell_mask,dtype=bool),step_size,smooth_tol,channel_axis)
    else:
        if pole_refinement == "contour":
            raise ValueError("contour pole_refinement requires channel_axis")
        mesh,radii,mode=_dt_medial_axis_impl(cell_mask,step_size=step_size,
            smooth_tol=smooth_tol,return_geometry=True,spine_mode=spine_mode,
            channel_axis=channel_axis)
        diagnostic=dict(method='legacy_contour',status='not_applicable',reason='',poles=[])
    if return_diagnostics:return mesh,radii,mode,diagnostic
    return _dt_result(mesh,radii,mode,return_geometry)


def _channel_raw_contour(mask):
    """Closed raw marching-squares contour in original image coordinates."""
    contours = find_contours(np.pad(mask, 1).astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)[:, ::-1] - 1.0
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    return np.ascontiguousarray(contour, dtype=float)


@njit(nogil=True, cache=True)
def _channel_point_inside_numba(x, y, contour):
    count = 0
    for j in range(len(contour) - 1):
        ax, ay = contour[j, 0], contour[j, 1]
        bx, by = contour[j + 1, 0], contour[j + 1, 1]
        crosses = (ay > y) != (by > y)
        dy = by - ay
        hit_x = ax + (y - ay) * (bx - ax) / (dy if dy != 0. else 1.)
        if crosses and x < hit_x:
            count += 1
    return count % 2 == 1


@njit(nogil=True, cache=True)
def _channel_points_inside_numba(points, contour):
    result = np.empty(len(points), dtype=np.bool_)
    for i in range(len(points)):
        result[i] = _channel_point_inside_numba(points[i, 0], points[i, 1], contour)
    return result


@njit(nogil=True, cache=True)
def _channel_point_segment_distance_numba(x, y, a, b):
    best = np.inf
    for j in range(len(a)):
        vx, vy = b[j, 0] - a[j, 0], b[j, 1] - a[j, 1]
        den = vx * vx + vy * vy
        dx, dy = x - a[j, 0], y - a[j, 1]
        t = (dx * vx + dy * vy) / (den if den > 0. else 1.)
        if t < 0.:
            t = 0.
        elif t > 1.:
            t = 1.
        nx, ny = a[j, 0] + t * vx, a[j, 1] + t * vy
        ex, ey = x - nx, y - ny
        squared = ex * ex + ey * ey
        # np.min propagates NaN; an ordinary comparison would silently ignore it.
        if np.isnan(squared):
            return np.nan
        if squared < best:
            best = squared
    return np.sqrt(best)


@njit(nogil=True, cache=True)
def _channel_segment_distances_numba(points, a, b):
    result = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        result[i] = _channel_point_segment_distance_numba(points[i, 0], points[i, 1], a, b)
    return result


@njit(nogil=True, cache=True)
def _channel_outside_distance_numba(points, contour):
    result = np.zeros(len(points), dtype=np.float64)
    for i in range(len(points)):
        if not _channel_point_inside_numba(points[i, 0], points[i, 1], contour):
            result[i] = _channel_point_segment_distance_numba(
                points[i, 0], points[i, 1], contour[:-1], contour[1:])
    return result


def _channel_points_inside(points, contour):
    """Even-odd polygon test, vectorized over candidate cap samples."""
    points = np.atleast_2d(points)
    if points.dtype == np.float64 and contour.dtype == np.float64:
        return _channel_points_inside_numba(points, contour)
    a, b = contour[:-1], contour[1:]
    x, y = points[:, 0:1], points[:, 1:2]
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    dy = b[:, 1] - a[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_x = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / np.where(dy != 0, dy, 1.)
    return np.count_nonzero(crosses & (x < hit_x), axis=1) % 2 == 1


def _channel_contour_distances(points, contour):
    points = np.atleast_2d(points)
    if points.dtype == np.float64 and contour.dtype == np.float64 and len(contour) > 1:
        return _channel_segment_distances_numba(points, contour[:-1], contour[1:])
    a = contour[:-1]
    v = contour[1:] - a
    den = np.sum(v * v, axis=1)
    d = points[:, None, :] - a[None, :, :]
    t = np.sum(d * v[None, :, :], axis=2) / np.where(den > 0, den, 1.)
    nearest = a[None, :, :] + np.clip(t, 0., 1.)[:, :, None] * v[None, :, :]
    return np.sqrt(np.min(np.sum((points[:, None, :] - nearest) ** 2, axis=2), axis=1))


def _channel_outside_distance(points, contour):
    points_array = np.asarray(points)
    if points_array.ndim == 2 and points_array.dtype == np.float64 and contour.dtype == np.float64 and len(contour) > 1:
        return _channel_outside_distance_numba(points_array, contour)
    inside = _channel_points_inside(points, contour)
    result = np.zeros(len(inside), dtype=float)
    if not np.all(inside):
        result[~inside] = _channel_contour_distances(np.asarray(points)[~inside], contour)
    return result


def _channel_first_exit(anchor, direction, contour):
    """First genuine inside-to-outside ray crossing, never a farther branch."""
    if not _channel_points_inside(np.asarray(anchor).reshape(1, 2), contour)[0]:
        return None
    a = contour[:-1]; v = contour[1:] - a
    den = direction[0] * v[:, 1] - direction[1] * v[:, 0]
    rel = a - anchor
    ok = np.abs(den) > 1e-12
    t = np.full(len(a), np.nan); s = t.copy()
    t[ok] = (rel[ok, 0] * v[ok, 1] - rel[ok, 1] * v[ok, 0]) / den[ok]
    s[ok] = (rel[ok, 0] * direction[1] - rel[ok, 1] * direction[0]) / den[ok]
    hits = np.sort(t[ok & (t > 1e-8) & (s >= -1e-9) & (s <= 1.+1e-9)])
    previous = -np.inf
    for value in hits:
        if value - previous < 1e-7:
            continue
        previous = value
        probes = anchor + np.asarray([value-1e-5, value+1e-5])[:, None] * direction
        membership = _channel_points_inside(probes, contour)
        if membership[0] and not membership[1]:
            return float(value)
    return None

def _channel_contour_radius(mesh, contour):
    """Robust shared body halfwidth, fixed for both cap-center fits."""
    n=len(mesh);lo=int(np.floor(.2*n));hi=max(lo+1,int(np.ceil(.8*n)))
    indices=np.unique(np.linspace(lo,hi-1,min(21,hi-lo)).astype(int))
    widths=[]
    for i in indices:
        rib=mesh[i];center=(rib[:2]+rib[2:])/2.;normal=rib[:2]-rib[2:]
        norm=float(np.linalg.norm(normal))
        if norm<1e-8:continue
        normal=normal/norm
        left=_channel_first_exit(center,normal,contour)
        right=_channel_first_exit(center,-normal,contour)
        if left is not None and right is not None and min(left,right)>0:
            widths.append((left+right)/2.)
    return float(np.median(widths)) if widths else 0.


def _channel_spline_model(centers, radius):
    """Centered spline on the selected route with bounded tangent extensions."""
    from scipy.interpolate import CubicSpline
    origin=centers[0].copy();points=centers-origin
    if not np.all(np.isfinite(points)) or len(points)<3:return None
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    smoothed=_spline_smooth(points,DEFAULT_SMOOTH_TOL) if len(points)>=4 else points.copy()
    # Constrain the old smoothing spline's endpoint motion to a radius corridor.
    if np.max(np.linalg.norm(smoothed-points,axis=1))>radius:
        return None
    spline=CubicSpline(arc,smoothed,bc_type='natural',axis=0)
    length=float(arc[-1])
    # The spline is bounded at its knots and at densely sampled continuous
    # route positions; linear extensions do not polynomially extrapolate.
    dense_s=np.unique(np.r_[arc,np.linspace(0.,length,max(101,4*len(points)))])
    dense=spline(dense_s)
    dense_distance=_channel_segment_distances(dense,points[:-1],points[1:])
    if np.max(dense_distance)>radius:return None
    first=spline(0.);last=spline(length)
    t0=spline(0.,1);t1=spline(length,1)
    if min(np.linalg.norm(t0),np.linalg.norm(t1))<1e-8:return None
    t0=t0/np.linalg.norm(t0);t1=t1/np.linalg.norm(t1)
    def evaluate(s,derivative=False):
        values=np.atleast_1d(s).astype(float)
        clipped=np.clip(values,0.,length)
        result=spline(clipped,1 if derivative else 0)
        below=values<0.;above=values>length
        if derivative:
            result[below]=t0;result[above]=t1
        else:
            result[below]=first+values[below,None]*t0
            result[above]=last+(values[above,None]-length)*t1
            result=result+origin
        return result[0] if np.ndim(s)==0 else result
    return evaluate,length,float(np.max(np.linalg.norm(smoothed-points,axis=1)))


def _copy_poles(poles):
    import copy
    return copy.deepcopy(poles)


def _reference_pole_constraints(contour,poles,mesh,body_count):
    """Freeze ordered raw-contour arcs and local bounds at initial baseline poles."""
    centers=_mesh_centers(mesh)
    length=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum())
    radii=[float(p['radius_px']) for p in poles];total=length+sum(radii)
    edges=np.linalg.norm(np.diff(contour,axis=0),axis=1)
    contour_arc=np.r_[0.,np.cumsum(edges)];period=float(contour_arc[-1])
    regions=[];deltas=[];uncertainties=[]
    for pole,R in zip(poles,radii):
        tip=np.asarray(pole['tip_xy']);center=np.asarray(pole['center_xy']);u=np.asarray(pole['outward_xy'])
        position,_=_route_coordinates(tip[None,:],contour)
        mid=(contour_arc[:-1]+contour_arc[1:])/2.
        difference=np.abs(mid-position[0]);difference=np.minimum(difference,period-difference)
        forward=np.sum(((contour[:-1]+contour[1:])/2.-center)*u,axis=1)>=-1e-8
        eligible=(difference<=2.*R)&forward
        nearest=int(np.argmin(difference));eligible[nearest]=True
        n=len(edges);left=nearest
        for _ in range(n-1):
            previous=(left-1)%n
            if previous==nearest or not eligible[previous]:break
            left=previous
        indices=[left]
        for _ in range(n-1):
            following=(indices[-1]+1)%n
            if following==left or not eligible[following]:break
            indices.append(following)
        ids=np.asarray(indices,dtype=int)
        regions.append(np.column_stack([contour[ids],contour[ids+1]]).tolist())
        deltas.append(max(1.,min(.5*R,.1*length)))
        uncertainties.append(max(1.,.5*R))
    return dict(length_px=total,body_center_arc_px=length,endpoint_radii=radii,
        poles=_copy_poles(poles),body_rib_count=int(body_count),
        length_bounds_px=[max(0.,total-sum(uncertainties)),total+sum(uncertainties)],
        center_search_bounds_px=[[0.,deltas[0]],[length-deltas[1],length]],
        endpoint_inward_bounds_px=deltas,endpoint_tip_uncertainty_px=uncertainties,
        terminal_contour_segments=regions)


def _reference_acceptance(mesh,reference):
    if mesh is None:return dict(valid=False,reason='missing_mesh')
    centers=_mesh_centers(mesh);radii=reference['endpoint_radii']
    length=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum())+sum(radii)
    lo,hi=reference['length_bounds_px'];distances=[]
    for pole,c,j,R in zip(reference['poles'],centers[[0,-1]],[1,-2],radii):
        u=c-centers[j];norm=np.linalg.norm(u)
        if norm<=1e-12:return dict(valid=False,reason='zero_terminal_direction')
        distances.append(float(np.linalg.norm(c+R*u/norm-np.asarray(pole['tip_xy']))))
    reason='' if lo-1e-8<=length<=hi+1e-8 else 'reference_length_out_of_bounds'
    if any(d>limit+1e-8 for d,limit in zip(distances,reference['endpoint_tip_uncertainty_px'])):
        reason='reference_terminal_coverage_lost'
    return dict(valid=not reason,reason=reason,length_px=length,
        length_delta_px=length-reference['length_px'],tip_displacements_px=distances)


def _channel_fit_fixed_cap(model, radius, contour, start, bounds=None, segments=None, reference_pole=None, max_tip_distance=None):
    """Optimize one scalar cap-center coordinate, never the shared radius."""
    from scipy.optimize import minimize_scalar
    evaluate,length,_=model
    lower,upper=((-radius,min(2.*radius,.45*length)) if start else
                 (max(.55*length,length-2.*radius),length+radius))
    if bounds is not None:lower,upper=map(float,bounds)
    endpoint=evaluate(0. if start else length)
    # Reference fixes contour indices and bounds before any center movement.
    a,b=contour[:-1],contour[1:]
    near=np.all((np.maximum(a,b)>=endpoint-4.*radius)&
                (np.minimum(a,b)<=endpoint+4.*radius),axis=1)
    if segments is None:segments=(a[near],b[near])
    if len(segments[0])<2:return None
    angles=np.linspace(-np.pi/2.,np.pi/2.,33)
    def fit_at(s,details=False):
        center=evaluate(float(s));tangent=evaluate(float(s),True)
        norm=float(np.linalg.norm(tangent))
        if norm<1e-8 or not _channel_points_inside(center[None,:],contour)[0]:
            return None if details else 1e12
        tangent=tangent/norm;outward=-tangent if start else tangent
        tip=center+radius*outward
        if reference_pole is not None and np.linalg.norm(tip-np.asarray(reference_pole['tip_xy']))>max_tip_distance:
            return None if details else 1e12
        normal=np.array([-outward[1],outward[0]])
        cap=center+radius*(np.cos(angles)[:,None]*outward+np.sin(angles)[:,None]*normal)
        distances=_channel_segment_distances(cap,*segments)
        outside=_channel_outside_distance(cap,contour)
        # Fixed-circle prior permits soft contour discrepancy; no variable radius.
        score=float(np.mean(distances**2)+np.mean(outside**2))
        if details:return dict(center_xy=center.tolist(),tip_xy=(center+radius*outward).tolist(),
            outward_xy=outward.tolist(),radius_px=float(radius),spine_parameter_px=float(s),
            cap_fit_score_px2=score,cap_mean_contour_distance_px=float(np.mean(distances)),
            cap_max_outside_px=float(np.max(outside)),status='refined',reason='')
        return score
    grid=np.linspace(lower,upper,25);scores=np.array([fit_at(s) for s in grid])
    index=int(np.argmin(scores))
    if scores[index]>=1e11:return None
    best=float(grid[index]);score=float(scores[index]);iterations=25
    lo=grid[max(0,index-1)];hi=grid[min(len(grid)-1,index+1)]
    if hi>lo:
        result=minimize_scalar(fit_at,bounds=(float(lo),float(hi)),method='bounded',
                               options={'xatol':.01,'maxiter':30})
        iterations+=int(result.nfev)
        if result.fun<score:best=float(result.x)
    info=fit_at(best,True);info['objective_evaluations']=iterations
    return info


def _channel_segment_distances(points,a,b):
    points=np.atleast_2d(points)
    if points.dtype == np.float64 and a.dtype == np.float64 and b.dtype == np.float64 and len(a):
        return _channel_segment_distances_numba(points,a,b)
    v=b-a;den=np.sum(v*v,axis=1)
    delta=points[:,None,:]-a[None,:,:]
    t=np.sum(delta*v[None,:,:],axis=2)/np.where(den>0.,den,1.)
    closest=a[None,:,:]+np.clip(t,0.,1.)[:,:,None]*v[None,:,:]
    return np.sqrt(np.min(np.sum((points[:,None,:]-closest)**2,axis=2),axis=1))


def _channel_guarded_spine(model,s0,s1,step,radius,return_tangents=False):
    """C1 Hermite end blending with exact adjacent-chord endpoint tangents."""
    evaluate,_,_=model
    span=s1-s0
    if span<=0.:return None
    start=evaluate(s0);end=evaluate(s1)
    t0=evaluate(s0,True);t1=evaluate(s1,True)
    t0=t0/np.linalg.norm(t0);t1=t1/np.linalg.norm(t1)
    guard=min(.1,.1*step,.02*span,.05*radius)
    blend=min(2.*step,.2*span)
    if blend<=guard:return None
    from scipy.interpolate import CubicHermiteSpline
    a=s0+blend;b=s1-blend
    left=CubicHermiteSpline([s0+guard,a],[start+guard*t0,evaluate(a)],
                            [t0,evaluate(a,True)],axis=0)
    right=CubicHermiteSpline([b,s1-guard],[evaluate(b),end-guard*t1],
                             [evaluate(b,True),t1],axis=0)
    def point(s,derivative=False):
        if s<=s0+guard:return t0 if derivative else start+(s-s0)*t0
        if s>=s1-guard:return t1 if derivative else end+(s-s1)*t1
        if s<a:return left(s,1 if derivative else 0)
        if s>b:return right(s,1 if derivative else 0)
        return evaluate(s,derivative)
    # Sample inside both Hermite spans, not just their endpoints.
    left_count=max(3,int(np.ceil((a-s0-guard)/(.25*step)))+1)
    right_count=max(3,int(np.ceil((s1-guard-b)/(.25*step)))+1)
    left_params=np.linspace(s0+guard,a,left_count)
    right_params=np.linspace(b,s1-guard,right_count)
    params=np.unique(np.r_[s0,left_params,np.linspace(a,b,max(2,int(np.ceil((b-a)/step))+1)),right_params,s1])
    points=np.array([point(s) for s in params])
    # Each segment is subdivided without shortcutting spline knots.
    result_params=[params[0]]
    for index,(p,q) in enumerate(zip(points[:-1],points[1:])):
        count=max(1,int(np.ceil(np.linalg.norm(q-p)/step)))
        result_params.extend(np.linspace(params[index],params[index+1],count+1)[1:])
    result=np.array([point(s) for s in result_params])
    if return_tangents:
        tangents=np.array([point(s,True) for s in result_params])
        tangents/=np.linalg.norm(tangents,axis=1)[:,None]
        return result,tangents
    return result


def _body_median_halfwidth(mesh, cap_rows=True):
    body=mesh[1:-1] if cap_rows else mesh
    values=np.linalg.norm(body[:,:2]-body[:,2:],axis=1)/2.
    values=values[np.isfinite(values)&(values>0.)]
    return float(np.median(values)) if len(values) else 0.


def _route_coordinates(points, route):
    a=route[:-1];delta=np.diff(route,axis=0);length=np.linalg.norm(delta,axis=1)
    arc=np.r_[0.,np.cumsum(length)]
    offset=np.atleast_2d(points)[:,None,:]-a[None,:,:]
    fraction=np.clip(np.sum(offset*delta[None,:,:],axis=2)/np.maximum(length**2,1e-12),0.,1.)
    projection=a[None,:,:]+fraction[:,:,None]*delta[None,:,:]
    distance=np.linalg.norm(np.atleast_2d(points)[:,None,:]-projection,axis=2)
    index=np.argmin(distance,axis=1);row=np.arange(len(index))
    return arc[index]+fraction[row,index]*length[index],distance[row,index]


def _channel_route_supported(points,route,radius,step):
    along,distance=_route_coordinates(points,route)
    return (np.all(np.isfinite(points)) and np.max(distance)<=max(1.,radius)
            and np.all(np.diff(along)>=-max(.25,.5*step)))


def _rib_at(center,tangent,radius,reference):
    tangent=tangent/np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]])
    if np.dot(normal,reference[:2]-reference[2:])<0:normal=-normal
    return np.r_[center+radius*normal,center-radius*normal]


def _terminal_mesh_from_poles(mesh,radius,poles,step):
    """Keep a contiguous exact body slice; only cap-covered end rows may go."""
    centers=_mesh_centers(mesh)
    r0,r1=(float(radius),float(radius)) if np.ndim(radius)==0 else tuple(float(r) for r in radius)
    c0,c1=[np.asarray(p['center_xy']) for p in poles]
    u0,u1=[np.asarray(p['outward_xy']) for p in poles]
    gap=min(.1,.05*step,.02*min(r0,r1))
    inside=(np.sum((centers-c0)*u0,axis=1)<-gap)&(np.sum((centers-c1)*u1,axis=1)<-gap)
    inside[[0,-1]]=False
    indices=np.flatnonzero(inside)
    if len(indices)==0:return None,'no_body_between_poles',{}
    first,last=int(indices[0]),int(indices[-1])
    if not np.all(inside[first:last+1]):return None,'nonterminal_cap_conflict',{}
    cap0=_rib_at(c0,-u0,r0,mesh[first]);cap1=_rib_at(c1,u1,r1,mesh[last])
    first,last,conflict=terminal_body_indices(mesh,cap0,cap1,first,last)
    if conflict==1:return None,'terminal_ribs_all_redundant',{}
    if conflict==2:return None,'nonterminal_rib_crosses_cap',{}
    # A short parallel diameter fixes the actual adjacent-center profile tangent.
    guard=min(.1,.1*step,.05*min(r0,r1),.1*np.linalg.norm(c1-c0))
    if guard<=1e-5:return None,'zero_terminal_connection',{}
    guard0=_rib_at(c0-guard*u0,-u0,r0,mesh[first])
    guard1=_rib_at(c1-guard*u1,u1,r1,mesh[last])
    # A rotated final body rib can clear the diameter but cross its tangent guard.
    # Remove only a terminal prefix/suffix until both fixed terminal rows are clear.
    first,last,guard_conflict=terminal_body_indices(mesh,guard0,guard1,first,last)
    if guard_conflict==1:return None,'terminal_ribs_all_redundant',{}
    if guard_conflict==2:return None,'nonterminal_rib_crosses_guard',{}
    def quad_area(a,b):
        return _signed_polygon_area(np.array([a[:2],b[:2],b[2:],a[2:]]))
    expected=quad_area(cap0,guard0)
    # Keep terminal connecting quadrilaterals oriented like the fixed cap rows.
    # A rib may miss both lines yet fold the side polygon at its connection.
    while first<=last and (abs(quad_area(guard0,mesh[first]))<1e-3 or quad_area(guard0,mesh[first])*expected<=0.):first+=1
    while last>=first and (abs(quad_area(mesh[last],guard1))<1e-3 or quad_area(mesh[last],guard1)*expected<=0.):last-=1
    if first>last:return None,'terminal_ribs_all_redundant',{}
    candidate=np.vstack([cap0,guard0,mesh[first:last+1],guard1,cap1])
    if not _validate_non_crossing_mesh(candidate):return None,'terminal_mesh_invalid',{}
    final=_mesh_centers(candidate)
    for record,c,u,r in zip(poles,final[[0,-1]],[(final[0]-final[1])/np.linalg.norm(final[0]-final[1]),
                                              (final[-1]-final[-2])/np.linalg.norm(final[-1]-final[-2])],[r0,r1]):
        record['final_tip_error_px']=float(np.linalg.norm(c+r*u-np.asarray(record['tip_xy'])))
        if record['final_tip_error_px']>1e-6:return None,'final_tip_inconsistent',{}
    details=dict(trimmed_ribs=int(len(mesh)-2-(last-first+1)),added_terminal_ribs=2,
        preserved_original_rows=[first,last],preserved_candidate_rows=[2,2+last-first])
    return candidate,'',details


def _baseline_poles_from_body(body,contour,step,independent_radii=False,stable_tangents=False):
    centers=_mesh_centers(body);radius=_body_median_halfwidth(body,cap_rows=False)
    radii=(radius,radius)
    local_count=None
    if independent_radii:
        local_count=min(len(body),max(3,min(21,int(np.ceil(.2*len(body))))))
        radii=tuple(_body_median_halfwidth(rows,cap_rows=False) for rows in (body[:local_count],body[-local_count:]))
    if min(radii)<=.25 or not np.all(np.isfinite(radii)):return None,None,'invalid_baseline_radius'
    poles=[]
    for start,radius in zip([True,False],radii):
        i=0 if start else -1
        window=min(5,len(centers)) if stable_tangents else 2
        j=window-1 if start else -window
        u=centers[i]-centers[j];norm=np.linalg.norm(u)
        if norm<=1e-12:return None,None,'invalid_baseline_tangent'
        u/=norm
        distance=_channel_first_exit(centers[i],u,contour)
        if distance is None or distance>4.*radius:return None,None,'baseline_pole_not_supported'
        tip=centers[i]+distance*u;center=tip-radius*u
        if not _channel_points_inside(center[None,:],contour)[0]:return None,None,'baseline_pole_center_outside'
        pole=dict(center_xy=center.tolist(),tip_xy=tip.tolist(),outward_xy=u.tolist(),radius_px=radius)
        if independent_radii or stable_tangents:
            normal=np.array([-u[1],u[0]]);angles=np.linspace(-np.pi/2.,np.pi/2.,33)
            arc=center+radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*normal)
            distance_to_contour=_channel_contour_distances(arc,contour)
            pole.update(cap_max_outside_px=float(np.max(_channel_outside_distance(arc,contour))),
                        cap_mean_contour_distance_px=float(np.mean(distance_to_contour)),
                        local_radius_rib_count=local_count)
        poles.append(pole)
    # Body has no cap rows yet: temporary copies isolate the all-body median.
    temporary=np.vstack([body[0],body,body[-1]])
    candidate,reason,details=_terminal_mesh_from_poles(temporary,radii if independent_radii else radius,poles,step)
    if candidate is None:return None,None,reason
    return candidate,radii,dict(poles=poles,local_radius_rib_count=local_count,**details)


def _initial_reference(seed,dt,contour,raw_contour,step,branched):
    radii=_sample_refinement_radii(seed,dt)
    body=_build_ribs_contour_constrained(seed,radii,contour)
    if body is None or not _validate_non_crossing_mesh(body,radii):
        return None,None,None,'initial_contour_body_invalid'
    mesh,ep,info=_baseline_poles_from_body(body,raw_contour,step,
        independent_radii=branched,stable_tangents=True)
    if mesh is None:return None,None,None,'initial_reference_'+str(info)
    if not np.all(_channel_points_inside(_mesh_centers(mesh),raw_contour)):
        return None,None,None,'initial_reference_center_outside'
    reference=_reference_pole_constraints(raw_contour,info['poles'],mesh,len(body))
    reference['radius_source']=('initial_local_terminal_body_median_halfwidth' if branched
        else 'initial_contour_body_median_halfwidth')
    reference['local_radius_rib_count']=info['local_radius_rib_count']
    return mesh,ep,reference,''


def _channel_body_refinement(seed,dt,contour,raw_contour,step,smooth_tol,branched=False,
                             reference=None,initial_mesh=None,initial_ep=None):
    if reference is None:
        initial_mesh,initial_ep,reference,reason=_initial_reference(seed,dt,contour,raw_contour,step,branched)
        if reference is None:
            return None,None,dict(iterations=0,max_iterations=MAX_BODY_ITERATIONS,converged=False,
                best_shift_px=None,stop_reason=reason,midpoint_updates=0)
    current=seed.copy();best=initial_mesh.copy();best_shift=np.inf;last_reason='iteration_limit'
    ep=tuple(reference['endpoint_radii']);poles=reference['poles']
    seed_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(seed,axis=0),axis=1))]
    seed_widths=_sample_dt(seed,dt)
    def route_supported(points):
        along,distance=_route_coordinates(points,seed)
        local=np.interp(along,seed_arc,seed_widths)
        return (np.all(np.isfinite(points)) and np.all(distance<=np.maximum(1.,local))
                and np.all(np.diff(along)>=-max(.25,.5*step)))
    diag=dict(iterations=0,max_iterations=MAX_BODY_ITERATIONS,converged=False,
        best_shift_px=None,stop_reason='iteration_limit',midpoint_updates=0,
        endpoint_neighborhoods_fixed=True,reference_length_px=reference['length_px'],
        radius_body_rib_count=reference['body_rib_count'],radius_population='initial_contour_body_before_iterations',
        endpoint_radii=list(ep),best_iteration=0,initial_reference_retained=True)
    if not branched:diag['shared_radius_px']=ep[0]
    else:diag['radius_local_rib_count']=reference['local_radius_rib_count']
    for iteration in range(MAX_BODY_ITERATIONS):
        diag['iterations']=iteration+1
        sampled=_sample_refinement_radii(current,dt)
        body=_build_ribs_contour_constrained(current,sampled,contour)
        if body is None:last_reason='contour_rib_construction_failed';break
        observed=_mesh_centers(body);shift=float(_mean_centerline_shift(current,observed))
        valid=_validate_non_crossing_mesh(body,sampled) and route_supported(observed)
        candidate=None
        if valid:
            temporary=np.vstack([body[0],body,body[-1]])
            candidate,reason,details=_terminal_mesh_from_poles(temporary,ep,_copy_poles(poles),step)
            if candidate is not None:
                acceptance=_reference_acceptance(candidate,reference)
                if not acceptance['valid']:candidate=None;reason=acceptance['reason']
            if candidate is None:last_reason=reason
            elif shift<best_shift:
                best=candidate;best_shift=shift;diag.update(best_shift_px=shift,best_iteration=iteration+1,
                    initial_reference_retained=False,acceptance=acceptance)
            if candidate is not None and shift<BODY_CONVERGENCE_TOL_PX:
                diag['converged']=True;diag['stop_reason']='converged';break
        next_cl=_damped_centerline_update(current,observed,step,smooth_tol,
            radii=sampled,update_beta=BODY_UPDATE_FRACTION,fixed_endpoints=True)
        diag['midpoint_updates']+=1
        if next_cl is None:last_reason='spine_midpoint_update_failed';break
        if not route_supported(next_cl):last_reason='spine_left_selected_route';break
        if not np.allclose(next_cl[[0,-1]],seed[[0,-1]],rtol=0.,atol=1e-8):
            last_reason='seed_endpoint_neighborhood_lost';break
        current=next_cl
    if not diag['converged'] and diag['iterations']<MAX_BODY_ITERATIONS:diag['stop_reason']=last_reason
    return best,ep,diag


def _baseline_spine_model(centers):
    from scipy.interpolate import CubicSpline
    lengths=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    if np.any(lengths<=1e-8) or not np.all(np.isfinite(centers)):return None
    arc=np.r_[0.,np.cumsum(lengths)];origin=centers[0].copy()
    # Clamp endpoint derivatives to the actual baseline cap guards.
    t0=(centers[1]-centers[0])/lengths[0];t1=(centers[-1]-centers[-2])/lengths[-1]
    spline=CubicSpline(arc,centers-origin,axis=0,bc_type=((1,t0),(1,t1)))
    length=float(arc[-1])
    def evaluate(s,derivative=False):
        values=np.atleast_1d(s).astype(float);inside=np.clip(values,0.,length)
        result=spline(inside,1 if derivative else 0);low=values<0.;high=values>length
        if derivative:result[low]=t0;result[high]=t1
        else:
            result[low]=values[low,None]*t0
            result[high]=centers[-1]-origin+(values[high,None]-length)*t1
            result+=origin
        return result[0] if np.ndim(s)==0 else result
    return evaluate,length,0.


def _refine_channel_terminal_poles(mesh,mask,step_size,reference=None):
    """Bound both centers inward and score against immutable initial terminal arcs."""
    step=1. if step_size is None else float(step_size);contour=_channel_raw_contour(mask)
    def reject(reason,**extra):return None,None,dict(status='rejected',reason=reason,**extra)
    if reference is None:return reject('missing_initial_reference')
    radii=tuple(reference['endpoint_radii']);centers=_mesh_centers(mesh)
    model=_baseline_spine_model(centers)
    if model is None:return reject('invalid_pole_model')
    length=model[1];deltas=reference['endpoint_inward_bounds_px']
    bounds=[(0.,min(deltas[0],.45*length)),(max(.55*length,length-deltas[1]),length)]
    poles=[]
    for i,start in enumerate([True,False]):
        local=np.asarray(reference['terminal_contour_segments'][i],dtype=float)
        info=_channel_fit_fixed_cap(model,radii[i],contour,start,bounds=bounds[i],
            segments=(local[:,:2],local[:,2:]),reference_pole=reference['poles'][i],
            max_tip_distance=reference['endpoint_tip_uncertainty_px'][i])
        if info is None:return reject('no_supported_local_pole_center')
        parameter=float(info['spine_parameter_px'])
        if not bounds[i][0]-1e-8<=parameter<=bounds[i][1]+1e-8:
            return reject('pole_center_search_bound_violation')
        reference_center=np.asarray(reference['poles'][i]['center_xy'])
        inward=-np.asarray(reference['poles'][i]['outward_xy'])
        inward_shift=float((np.asarray(info['center_xy'])-reference_center)@inward)
        if inward_shift < -1e-6 or inward_shift > deltas[i]+1e-6:
            return reject('pole_center_terminal_neighborhood_lost')
        info['center_inward_displacement_px']=inward_shift
        info['search_bounds_px']=list(bounds[i]);info['terminal_contour_fixed']=True;poles.append(info)
    if poles[0]['spine_parameter_px']>=poles[1]['spine_parameter_px']:
        return reject('unordered_cap_centers',poles=poles)
    candidate,reason,details=_terminal_mesh_from_poles(mesh,radii,poles,step)
    if candidate is None:return reject(reason,poles=poles)
    acceptance=_reference_acceptance(candidate,reference)
    if not acceptance['valid']:return reject(acceptance['reason'],poles=poles,acceptance=acceptance)
    if np.max(_channel_outside_distance(_mesh_centers(candidate),contour))>1e-6:
        return reject('terminal_center_outside',poles=poles)
    return candidate,radii,dict(status='refined',reason='',poles=poles,
        shared_radius=radii[0]==radii[1],endpoint_radii=list(radii),
        radius_source=reference['radius_source'],pole_optimizer_applied=True,
        fit_low_confidence=any(p['cap_max_outside_px']>1. for p in poles),
        acceptance=acceptance,fit_center_arc_px=length,fit_input_ribs=len(mesh),**details)


def _channel_contour_pipeline(mask,step_size,smooth_tol,channel_axis):
    axis=_normalize_channel_axis(channel_axis)
    # Compute once in canonical axis direction, then reverse output order.
    # Contour candidate tie-breaking must not change geometry under reversal.
    if axis[int(np.argmax(np.abs(axis)))]<0:
        mesh,ep,mode,d=_channel_contour_pipeline(mask,step_size,smooth_tol,-axis)
        if mesh is not None:
            mesh=mesh[::-1][:,[2,3,0,1]].copy();ep=tuple(ep[::-1])
            d['poles']=d.get('poles',[])[::-1]
            baseline=d.get('pole_baseline',{})
            if baseline.get('centers_xy'):baseline['centers_xy']=baseline['centers_xy'][::-1]
            if baseline.get('endpoint_radii'):baseline['endpoint_radii']=baseline['endpoint_radii'][::-1]
            if d.get('endpoint_radii'):d['endpoint_radii']=d['endpoint_radii'][::-1]
            body=d.get('body_refinement',{})
            if body.get('endpoint_radii'):body['endpoint_radii']=body['endpoint_radii'][::-1]
            for record in (d.get('acceptance',{}),body.get('acceptance',{})):
                if 'tip_displacements_px' in record:record['tip_displacements_px']=record['tip_displacements_px'][::-1]
            fitted_arc=d.get('fit_center_arc_px')
            if fitted_arc is not None:
                for pole in d.get('poles',[]):
                    if 'spine_parameter_px' in pole:pole['spine_parameter_px']=fitted_arc-pole['spine_parameter_px']
                    if 'search_bounds_px' in pole:
                        a,b=pole['search_bounds_px'];pole['search_bounds_px']=[fitted_arc-b,fitted_arc-a]
            reference=d.get('reference_geometry')
            if reference:
                for key in ('endpoint_radii','poles','endpoint_inward_bounds_px','endpoint_tip_uncertainty_px','terminal_contour_segments'):
                    reference[key]=reference[key][::-1]
                L=reference['body_center_arc_px']
                reference['center_search_bounds_px']=[[L-b,L-a] for a,b in reference['center_search_bounds_px'][::-1]]
            if 'preserved_original_rows' in d:
                a,z=d['preserved_original_rows'];n=d.get('fit_input_ribs',baseline['n_ribs'])
                d['preserved_original_rows']=[n-1-z,n-1-a]
                a,z=d['preserved_candidate_rows'];n=len(mesh)
                d['preserved_candidate_rows']=[n-1-z,n-1-a]
        return mesh,ep,mode,d
    step=1. if step_size is None else float(step_size)
    diagnostics=dict(method='contour_pipeline',version=CHANNEL_POLE_REFINEMENT_VERSION,
        status='rejected',reason='',poles=[],trimmed_ribs=0,
        body_refinement=dict(iterations=0,max_iterations=MAX_BODY_ITERATIONS,converged=False,
                             best_shift_px=None,stop_reason='not_started',midpoint_updates=0),pole_baseline={},
        branch_info=dict(is_branched=False),branch_policy='shared_radius_rod')
    def reject(reason):
        diagnostics['reason']=reason
        return None,None,'invalid',diagnostics
    if mask.ndim!=2 or mask.size==0 or mask.sum()<4:return reject('invalid_mask')
    if not np.isfinite(step) or step<=0:return reject('invalid_step')
    # Fixed explicit padding makes all contours, DTs and skeletons crop invariant.
    pad=4
    ys,xs=np.where(mask);origin=np.array([int(xs.min()),int(ys.min())],dtype=float)
    tight=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    working=np.pad(tight,pad)
    output_shift=np.tile(origin-pad,2)
    raw_contours=find_contours(working.astype(float),.5)
    if len(raw_contours)!=1:return reject('unsupported_raw_contour_topology')
    raw_contour=_close_contour(raw_contours[0][:,::-1]);raw_dt=distance_transform_edt(working)
    if _is_channel_round_cell(working):
        mesh,ep=_fallback_round_cell(working,raw_dt)
        if mesh is None:return reject('round_geometry_failed')
        diagnostics.update(status='not_applicable',reason='dedicated_round')
        return mesh+output_shift,ep,'round',diagnostics
    smooth=gaussian_filter(working.astype(float),sigma=.8,mode='constant',cval=0.)>.5
    dt=distance_transform_edt(smooth);skeleton=skeletonize(smooth)
    selected,_=_extract_channel_axis_path(skeleton,axis)
    if selected is None:
        points=np.argwhere(skeleton)[:,::-1]
        if len(points)==2:
            delta=points[1]-points[0]
            if np.max(np.abs(delta))>1 or abs(float(delta@axis))<=1e-12:
                return reject('no_channel_axis_path')
            if float(delta@axis)<0:points=points[::-1].copy()
            mesh,ep=_fallback_two_point_cell(points,working,raw_dt)
            if mesh is not None:
                diagnostics.update(status='not_applicable',reason='dedicated_short_capsule')
                return mesh+output_shift,ep,'short_capsule',diagnostics
        return reject('no_channel_axis_path')
    branch=_primary_branch_info(skeleton,selected,raw_dt)
    if branch is not None:
        diagnostics['branch_info']=dict(is_branched=True,unused_arm_length_px=branch['unused_arm_length_px'],
            median_radius_px=branch['median_radius_px'],threshold_radius_factor=2.)
        diagnostics['branch_policy']='independent_local_baseline'
    route=np.asarray(selected,dtype=float)
    seed=_moving_average(route,CHANNEL_AXIS_SMOOTH_WINDOW)
    seed[0]=route[0];seed[-1]=route[-1]
    supported=_interp_equal_steps(seed,step)
    if smooth_tol is not None and len(seed)>=4:seed=_spline_smooth(seed,smooth_tol)
    seed=_interp_equal_steps(seed,step)
    contours=find_contours(smooth.astype(float),.5)
    if not contours:return reject('no_smooth_contour')
    contour=_close_contour(max(contours,key=len)[:,::-1])
    seed=_guard_spline_seed_endpoints(seed,supported,dt,contour)
    initial,initial_ep,reference,reason=_initial_reference(seed,dt,contour,raw_contour,step,branch is not None)
    if initial is None:return reject(reason)
    baseline,ep,body_diag=_channel_body_refinement(seed,dt,contour,raw_contour,step,smooth_tol,
        branched=branch is not None,reference=reference,initial_mesh=initial,initial_ep=initial_ep)
    diagnostics['body_refinement']=body_diag
    def shift_reference(ref,translation):
        import copy
        result=copy.deepcopy(ref)
        for pole in result['poles']:
            for key in ('center_xy','tip_xy'):pole[key]=(np.asarray(pole[key])+translation).tolist()
        result['terminal_contour_segments']=[(np.asarray(rows)+np.tile(translation,2)).tolist()
            for rows in result['terminal_contour_segments']]
        return result
    diagnostics['reference_geometry']=shift_reference(reference,origin-pad)
    diagnostics['pole_baseline']=dict(endpoint_radii=list(initial_ep),n_ribs=len(initial),
        length_px=reference['length_px'],centers_xy=[p['center_xy'] for p in diagnostics['reference_geometry']['poles']],
        radius_source=reference['radius_source'])
    if branch is None:diagnostics['pole_baseline']['radius_px']=initial_ep[0]
    else:diagnostics['branch_policy']='independent_local_radius_bounded_fit'
    if baseline is None:baseline=initial.copy()
    candidate,candidate_ep,result=_refine_channel_terminal_poles(baseline.copy(),working,step,reference=reference)
    if candidate is None:
        diagnostics.update(status='baseline_fallback',reason=result.get('reason','local_pole_refinement_failed'),
            poles=_copy_poles(diagnostics['reference_geometry']['poles']),trimmed_ribs=0,
            endpoint_radii=list(initial_ep),shared_radius=initial_ep[0]==initial_ep[1],
            pole_optimizer_applied=True,fallback_target='initial_reference',
            acceptance=_reference_acceptance(initial,reference),
            fit_low_confidence=any(p.get('cap_max_outside_px',0.)>1. for p in reference['poles']))
        return initial+output_shift,initial_ep,'channel_axis',diagnostics
    for pole in result['poles']:
        for key in ('center_xy','tip_xy'):pole[key]=(np.asarray(pole[key])+origin-pad).tolist()
    diagnostics.update(result)
    return candidate+output_shift,candidate_ep,'channel_axis',diagnostics

def _dt_medial_axis_impl(cell_mask, step_size=DEFAULT_STEP_SIZE,
                   smooth_tol=DEFAULT_SMOOTH_TOL,
                   return_geometry=False, spine_mode="default",
                   channel_axis=(0.0, 1.0)):
    """Compute cell mesh via distance-transform medial axis.

    Returns (mesh, endpoint_radii). When ``return_geometry`` is true, returns
    (mesh, endpoint_radii, geometry_mode). Cell length is computed as:
        length = spine_length + endpoint_radii[0] + endpoint_radii[-1]

    Parameters
    ----------
    cell_mask : np.ndarray, 2D bool
    step_size : float or None
    smooth_tol : float
    spine_mode : {"default", "channel_axis"}
        channel_axis keeps the skeleton route with maximum axial endpoint
        separation, lightly smooths and evenly resamples it, and uses raw-DT
        rib half-widths and caps (no neighbour clipping).
        Default production behavior remains unchanged.
    channel_axis : pair of float
        Image [x,y] direction, normalized internally; default (0,1).
    return_geometry : bool
        Include the explicit reconstruction mode: ``rod``,
        ``short_capsule``, ``round``, ``branched_primary``, or ``invalid``.
        ``channel_axis`` is the Moma worker default; the standalone core keeps
        ``spine_mode="default"`` for legacy reproduction. Channel profiles use
        saved axis path order without asserting biological pole polarity.

    Returns
    -------
    (mesh, endpoint_radii) or (None, None); optionally with geometry mode.
    """

    if spine_mode not in {"default", "channel_axis"}:
        raise ValueError("spine_mode must be default or channel_axis")
    axis = _normalize_channel_axis(channel_axis) if spine_mode == "channel_axis" else None
    mask = cell_mask.astype(bool)
    if mask.sum() < 4:
        return _dt_result(None, None, 'invalid', return_geometry)

    # Marching squares needs background outside the cell to close its contour.
    # Four pixels also contain the sigma=0.8 Gaussian kernel used below.
    if (np.any(mask[0]) or np.any(mask[-1]) or
            np.any(mask[:, 0]) or np.any(mask[:, -1])):
        pad = 4
        mesh, ep, mode = _dt_medial_axis_impl(
            np.pad(mask, pad), step_size=step_size, smooth_tol=smooth_tol,
            return_geometry=True, spine_mode=spine_mode, channel_axis=channel_axis)
        if mesh is not None:
            mesh = mesh - pad
        return _dt_result(mesh, ep, mode, return_geometry)

    dt_raw = distance_transform_edt(mask)
    round_eligible = (_is_channel_round_cell(mask) if spine_mode == "channel_axis"
                      else _is_round_like_cell(mask))
    if round_eligible:
        mesh, ep = _fallback_round_cell(mask, dt_raw)
        if mesh is not None:
            return _dt_result(
                _enforce_monotonic_mesh(mesh), ep, 'round', return_geometry)
        return _dt_result(None, None, 'invalid', return_geometry)

    mask_smooth = gaussian_filter(
        mask.astype(float), sigma=0.8, mode='constant', cval=0.0) > 0.5
    dt = distance_transform_edt(mask_smooth)
    skel_bin = skeletonize(mask_smooth)

    if spine_mode == "channel_axis":
        selected, _ = _extract_channel_axis_path(skel_bin, axis)
        if selected is None:
            # Retain genuine tiny-cell semantics, without hiding a failed rod
            # route behind a different reconstruction. Loops and zero-span
            # paths with three or more skeleton pixels remain invalid.
            if int(skel_bin.sum()) <= 2:
                points = np.argwhere(skel_bin)
                if len(points) == 2:
                    delta = points[1] - points[0]
                    if np.max(np.abs(delta)) > 1 or abs(float(delta[::-1] @ axis)) <= 1e-12:
                        return _dt_result(None, None, "invalid", return_geometry)
                    cl2 = points[:, ::-1].copy()
                    mesh, ep = _fallback_two_point_cell(cl2, mask, dt_raw)
                    mode = "short_capsule" if _mesh_centerline_has_axis(mesh) else "round"
                else:
                    return _dt_result(None, None, "invalid", return_geometry)
                return _dt_result(mesh, ep, mode if mesh is not None else "invalid", return_geometry)
            return _dt_result(None, None, "invalid", return_geometry)
        mesh, ep = _build_channel_axis_mesh(selected, dt_raw, step_size)
        return _dt_result(mesh, ep, "channel_axis" if mesh is not None else "invalid", return_geometry)

    cl_pts = _extract_centerline(skel_bin)
    if cl_pts is None or len(cl_pts) < 3:
        # Small/round cells can have a 0/1-pixel skeleton; very short cells can
        # have exactly two valid DT medial points. Treat those as separate
        # geometry modes instead of failing the rod-cell pipeline.
        cl2 = _extract_two_point_centerline(skel_bin)
        if cl2 is not None:
            mesh, ep = _fallback_two_point_cell(cl2, mask, dt_raw)
            geometry_mode = (
                'short_capsule' if _mesh_centerline_has_axis(mesh) else 'round')
        else:
            mesh, ep = _fallback_round_cell(mask, dt_raw)
            geometry_mode = 'round'
        if mesh is not None:
            return _dt_result(
                _enforce_monotonic_mesh(mesh), ep, geometry_mode,
                return_geometry)
        return _dt_result(None, None, 'invalid', return_geometry)

    # A substantial unused arm needs an explicit primary-body representation.
    # Whole-contour recentering can switch a curved longest path into that arm.
    branch = _primary_branch_info(skel_bin, cl_pts, dt)
    if branch is not None:
        mesh, ep = _build_primary_branch_mesh(cl_pts, dt_raw, step_size)
        return _dt_result(mesh, ep,
                          'branched_primary' if mesh is not None else 'invalid',
                          return_geometry)

    # Smooth: moving average
    if len(cl_pts) >= 5:
        w = max(3, min(5, len(cl_pts) // 4 * 2 + 1))
        cl_ma = _moving_average(cl_pts, w)
        if not np.any(np.isnan(cl_ma)):
            cl_pts = cl_ma

    # Keep a mask-supported alternative if spline endpoint motion later prevents
    # the first contour ribs from being constructed.
    supported_cl_pts = cl_pts.copy()

    # Smooth: spline
    if smooth_tol is not None and len(cl_pts) >= 4:
        cl_sm = _spline_smooth(cl_pts, smooth_tol)
        if not np.any(np.isnan(cl_sm)):
            cl_pts = cl_sm

    # Interpolate to equal steps
    if step_size is None:
        xs, ys = cl_pts[:, 0], cl_pts[:, 1]
        size = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
        step_size = min(3.0, max(1.0, size / 7.5))

    cl = _interp_equal_steps(cl_pts, step_size)
    if cl is None or len(cl) < 3:
        return _dt_result(None, None, 'invalid', return_geometry)

    # Sample DT at centerline points -> rib radii
    radii = _sample_dt(cl, dt)
    if len(radii) >= 3:
        radii = _moving_average_1d(radii, max(3, len(radii) // 8 * 2 + 1))

    # ---- Build initial mesh from DT radii, used as a pole-fitting seed ----
    mesh = _build_ribs(cl, radii)
    if mesh is None or len(mesh) < 3:
        return _dt_result(None, None, 'invalid', return_geometry)

    # ---- Pole radius seed from median DT radius ----
    # The final pole radius is recomputed after contour-rib refinement as
    # the mean half-width of body ribs.
    r_pole = float(np.median(radii))
    endpoint_radii = (r_pole, r_pole)

    # ---- Build contour-constrained mesh and anchor poles to contour tips ----
    cnts = find_contours(mask_smooth.astype(float), level=0.5)
    raw_cnts = find_contours(mask.astype(float), level=0.5)
    cnt_xy = None
    if cnts:
        cnt_xy = _close_contour(max(cnts, key=len)[:, ::-1])  # -> [x, y]
        supported_cl = _interp_equal_steps(supported_cl_pts, step_size)
        cl = _guard_spline_seed_endpoints(cl, supported_cl, dt, cnt_xy)
        pole_cnt_xy = cnt_xy
        if raw_cnts:
            pole_cnt_xy = _close_contour(max(raw_cnts, key=len)[:, ::-1])

        integrated = _refine_mesh_with_integrated_poles(
            cl, dt, cnt_xy, pole_cnt_xy, step_size,
            smooth_tol=smooth_tol, max_refine_iter=5,
            spine_update_beta=0.45,
        )
        if integrated is None:
            retry_tol = None if smooth_tol is None else max(smooth_tol * 5.0, 0.05)
            integrated = _refine_mesh_with_integrated_poles(
                cl, dt, cnt_xy, pole_cnt_xy, step_size,
                smooth_tol=retry_tol, max_refine_iter=7,
                spine_update_beta=0.35,
            )
        if integrated is None:
            integrated = _refine_mesh_with_integrated_poles(
                cl, dt, cnt_xy, pole_cnt_xy, step_size,
                smooth_tol=smooth_tol, max_refine_iter=5,
                spine_update_beta=1.0,
            )
        if integrated is not None:
            mesh, r_pole = integrated
        else:
            refined = _refine_mesh_fixed_point(
                cl, dt, cnt_xy, step_size, smooth_tol=smooth_tol,
                max_refine_iter=5, spine_update_beta=0.45,
            )
            if refined is None:
                retry_tol = None if smooth_tol is None else max(smooth_tol * 5.0, 0.05)
                refined = _refine_mesh_fixed_point(
                    cl, dt, cnt_xy, step_size, smooth_tol=retry_tol,
                    max_refine_iter=7, spine_update_beta=0.35,
                )
            if refined is None:
                refined = _refine_mesh_fixed_point(
                    cl, dt, cnt_xy, step_size, smooth_tol=smooth_tol,
                    max_refine_iter=5, spine_update_beta=1.0,
                )
            if refined is None:
                return _dt_result(None, None, 'invalid', return_geometry)
            mesh, radii = refined
            r_pole = _estimate_pole_radius_from_mesh(mesh)
            anchored_mesh = _anchor_poles_to_contour_tips(mesh, pole_cnt_xy, r_pole)
            if anchored_mesh is not None and _validate_non_crossing_mesh(anchored_mesh):
                mesh = anchored_mesh
        endpoint_radii = (r_pole, r_pole)

    mesh = _enforce_monotonic_mesh(mesh)
    if not _validate_non_crossing_mesh(mesh):
        return _dt_result(None, None, 'invalid', return_geometry)
    return _dt_result(mesh, endpoint_radii, 'rod', return_geometry)


def _dt_result(mesh, endpoint_radii, geometry_mode, return_geometry):
    """Return DT results with optional geometry metadata."""
    if return_geometry:
        return mesh, endpoint_radii, geometry_mode
    return mesh, endpoint_radii


@njit(nogil=True, cache=True)
def _mesh_centerline_has_axis(mesh, tol=1e-12):
    """Return whether mesh rib centers span a nonzero spatial axis."""
    if mesh is None or len(mesh) < 2:
        return False
    x0 = (mesh[0, 0] + mesh[0, 2]) / 2.0
    y0 = (mesh[0, 1] + mesh[0, 3]) / 2.0
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return False
    has_axis = False
    for i in range(1, len(mesh)):
        x = (mesh[i, 0] + mesh[i, 2]) / 2.0
        y = (mesh[i, 1] + mesh[i, 3]) / 2.0
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        dx = x - x0
        dy = y - y0
        if np.sqrt(dx * dx + dy * dy) > tol:
            has_axis = True
    return has_axis


# =============================================================================
# 2. Centerline extraction & smoothing
# =============================================================================

def _extract_centerline(skel_bin):
    """Extract ordered [x,y] centerline from binary skeleton."""
    from scipy.ndimage import label, generate_binary_structure
    struct = generate_binary_structure(2, 2)
    labels, n_labels = label(skel_bin, structure=struct)
    best_xy, best_len = None, 0.0
    for lbl in range(1, n_labels + 1):
        pts_rc = np.argwhere(labels == lbl)
        if len(pts_rc) < 3:
            continue
        ordered = _order_points(pts_rc)
        if ordered is None or len(ordered) < 3:
            continue
        d = np.diff(ordered, axis=0)
        length = float(np.sum(np.sqrt((d * d).sum(axis=1))))
        if length > best_len:
            best_len = length
            best_xy = ordered[:, ::-1].copy()
    return best_xy


def _is_channel_round_cell(mask):
    """Keep compact filled round cells; reject thin/disconnected tiny fragments."""
    from scipy.ndimage import label, binary_fill_holes
    if label(mask, structure=np.ones((3,3), dtype=bool))[1] != 1:
        return False
    if not np.array_equal(binary_fill_holes(mask), mask) or not _is_round_like_cell(mask):
        return False
    y, x = np.where(mask)
    spans = np.array([np.ptp(x)+1, np.ptp(y)+1], dtype=float)
    return bool(spans.max()/spans.min() <= 1.35)


def _normalize_channel_axis(channel_axis):
    """Return a finite unit vector in image [x,y] coordinates."""
    axis = np.asarray(channel_axis, dtype=np.float64)
    if axis.shape != (2,) or not np.isfinite(axis).all():
        raise ValueError("channel_axis must contain two finite values in [x,y] order")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("channel_axis must be nonzero")
    return axis / norm


def _extract_channel_axis_path(skeleton, channel_axis=(0.0, 1.0)):
    """Select connected terminal points with maximum channel-axis separation.

    The returned graph route keeps its original vertices and runs toward the
    positive axis. Loops without degree-one endpoints have no declared path.
    """
    from scipy.ndimage import label, generate_binary_structure
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    axis = _normalize_channel_axis(channel_axis)
    components, count = label(skeleton, structure=generate_binary_structure(2, 2))
    best_key, best_path, best_info = None, None, None
    for component in range(1, count + 1):
        points_rc = np.argwhere(components == component)
        if len(points_rc) < 3:
            continue
        points = points_rc[:, ::-1]
        lookup = {tuple(p): i for i, p in enumerate(points_rc)}
        rows, cols, weights = [], [], []
        degree = np.zeros(len(points), dtype=int)
        for i, (y, x) in enumerate(points_rc):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    j = lookup.get((y + dy, x + dx))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        weights.append(float(np.hypot(dx, dy)))
                        degree[i] += 1
        endpoints = np.flatnonzero(degree == 1)
        if len(endpoints) < 2:
            continue
        graph = coo_matrix((weights, (rows, cols)), shape=(len(points), len(points))).tocsr()
        projected = points @ axis
        for position, start in enumerate(endpoints):
            distance, previous = dijkstra(graph, directed=False, indices=int(start), return_predecessors=True)
            for end in endpoints[position + 1:]:
                span = float(abs(projected[end] - projected[start]))
                if span <= 1e-12 or not np.isfinite(distance[end]):
                    continue
                key = (span, float(distance[end]))
                if best_key is not None and key <= best_key:
                    continue
                indices, current = [int(end)], int(end)
                while current != start:
                    current = int(previous[current])
                    if current < 0:
                        break
                    indices.append(current)
                if current != start:
                    continue
                path = points[indices[::-1]].copy()
                if path[0] @ axis > path[-1] @ axis:
                    path = path[::-1].copy()
                path_projection = path @ axis
                best_key, best_path = key, path
                best_info = dict(axis_xy=axis.tolist(), selected_endpoints_xy=path[[0,-1]].tolist(),
                    graph_length_px=float(distance[end]), endpoint_span_px=span,
                    path_span_px=float(np.ptp(path_projection)),
                    skeleton_span_px=float(np.ptp(projected)),
                    backtracking_px=float((np.abs(np.diff(path_projection)).sum() - span) / 2),
                    component_skeleton_pixels=len(points), total_skeleton_pixels=int(skeleton.sum()),
                    component_count=int(count), ignored_skeleton_pixels=int(skeleton.sum())-len(points),
                    skeleton_xy=points.tolist(), path_xy=path.tolist())
    return best_path, best_info


def _primary_branch_info(skeleton, path_xy, dt):
    """Identify a genuine unused arm, rather than short skeleton pole spurs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    points = np.argwhere(skeleton)
    if path_xy is None or len(points) <= len(path_xy):
        return None
    index = {tuple(point): i for i, point in enumerate(points)}
    sources = np.array([index[(int(p[1]), int(p[0]))] for p in path_xy], dtype=np.int32)
    source_set = set(int(i) for i in sources)
    rows, cols, weights = [], [], []
    degree = np.zeros(len(points), dtype=int)
    for i, (row, col) in enumerate(points):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index.get((row + dy, col + dx))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    weights.append(float(np.hypot(dx, dy)))
                    degree[i] += 1
    unused_endpoints = [i for i in range(len(points)) if degree[i] == 1 and i not in source_set]
    if not unused_endpoints or np.count_nonzero(degree == 1) < 3:
        return None
    graph = coo_matrix((weights, (rows, cols)), shape=(len(points), len(points))).tocsr()
    distance = dijkstra(graph, indices=sources, directed=False, min_only=True)
    connected_unused = [i for i in unused_endpoints if np.isfinite(distance[i])]
    if not connected_unused:
        return None
    off_length = max(float(distance[i]) for i in connected_unused)
    local_radius = float(np.median(dt[path_xy[:, 1], path_xy[:, 0]]))
    if off_length <= 2.0 * local_radius:
        return None
    return dict(unused_arm_length_px=off_length, median_radius_px=local_radius,
                selected_geodesic_length_px=float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()),
                selected_path_xy=path_xy.tolist(),
                unused_endpoints_xy=points[unused_endpoints, ::-1].tolist(),
                skeleton_xy=points[:, ::-1].tolist())


def _build_channel_axis_mesh(path_xy, dt_raw, step_size):
    """Evenly sampled channel-axis body with unclipped raw-DT ribs.

    The selected 8-connected skeleton route is a pixel staircase. Building ribs
    on its vertices gives alternating 0.707/1 px spacing, lengthens the arc and
    tilts the ribs relative to the local path. A light centered moving average
    (fixed endpoints) removes the stair steps without changing the selected
    route or its ends, and equal arc-length resampling gives even spacing.
    Each rib half-width is bilinearly sampled from the raw-mask distance
    transform. It estimates an inscribed radius in pixel-center coordinates;
    it is not a guarantee of containment in the continuous mask contour. Ribs
    are not clipped against neighbouring samples; the unchanged strict
    non-crossing gate rejects invalid meshes.
    """
    step = 1.0 if step_size is None else float(step_size)
    if not np.isfinite(step) or step <= 0 or len(path_xy) < 3:
        return None, None
    route = np.asarray(path_xy, dtype=np.float64)
    smooth = _moving_average(route, CHANNEL_AXIS_SMOOTH_WINDOW)
    smooth[0] = route[0]
    smooth[-1] = route[-1]
    centers = _interp_equal_steps(np.ascontiguousarray(smooth), step)
    if len(centers) < 3:
        return None, None
    normals = _centerline_normals(centers)
    radii = _sample_dt(centers, dt_raw)
    endpoint_radii = (float(radii[0]), float(radii[-1]))
    if normals is None or min(endpoint_radii) <= 0:
        return None, None
    mesh = np.column_stack([centers + normals * radii[:, None],
                            centers - normals * radii[:, None]])
    if not _validate_non_crossing_mesh(mesh, radii):
        return None, None
    return mesh, endpoint_radii


def _build_primary_branch_mesh(path_xy, dt_raw, step_size):
    """Preserve the complete selected geodesic, with a non-crossing DT tube.

    This models primary-body geometry only. Raw mask statistics stay with the
    caller. Segment subdivision never shortcuts a branch junction or reorders
    ribs by PCA, and no contour fixed-point update can switch the chosen arms.
    """
    step = 1.0 if step_size is None else float(step_size)
    if not np.isfinite(step) or step <= 0 or len(path_xy) < 3:
        return None, None
    points = []
    for start, end in zip(path_xy[:-1], path_xy[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
        for j in range(n):
            points.append(start + (end - start) * (j / n))
    points.append(path_xy[-1].astype(float))
    centers = np.asarray(points, dtype=np.float64)
    normals = _centerline_normals(centers)
    radii = _sample_dt(centers, dt_raw)
    endpoint_radii = (float(radii[0]), float(radii[-1]))
    if normals is None or radii is None or min(endpoint_radii) <= 0:
        return None, None
    # A point on rib i must remain inside i's Voronoi half-plane against every
    # other path sample. This bounds thick ribs near a sharp turn or another arm.
    for i in range(len(centers)):
        delta = centers - centers[i]
        projection = np.abs(delta @ normals[i])
        relevant = projection > 1e-12
        relevant[i] = False
        if np.any(relevant):
            bounds = np.sum(delta[relevant] ** 2, axis=1) / (2.0 * projection[relevant])
            radii[i] = min(radii[i], float(bounds.min()) * (1.0 - 1e-6))
    mesh = np.column_stack([centers + normals * radii[:, None],
                            centers - normals * radii[:, None]])
    if not _validate_non_crossing_mesh(mesh, radii):
        return None, None
    return mesh, endpoint_radii


@njit(nogil=True, cache=True)
def _is_round_like_cell(mask, max_area=180, max_aspect=1.35):
    """Detect small near-isotropic masks that are better modeled as round cells."""
    ys, xs = np.where(mask)
    area = len(xs)
    if area < 4 or area > max_area:
        return False
    if area < 12:
        return True

    xs_float = xs.astype(np.float64)
    ys_float = ys.astype(np.float64)
    pts = np.column_stack((xs_float, ys_float))
    mean = np.array((np.mean(xs_float), np.mean(ys_float)))
    centered = pts - mean
    cov = centered.T @ centered / max(area - 1, 1)
    vals = np.linalg.eigvalsh(cov)
    if vals[0] <= 1e-12:
        return False
    aspect = float(np.sqrt(vals[1] / vals[0]))
    return aspect <= max_aspect


@njit(nogil=True, cache=True)
def _extract_two_point_centerline(skel_bin):
    """Return a 2-point [x,y] centerline if the skeleton has two medial points."""
    pts = np.argwhere(skel_bin)  # [row, col]
    if len(pts) < 2:
        return None

    # Use the two farthest skeleton pixels. For exactly two points this is just
    # the skeleton. For fragmented tiny skeletons it gives the most stable axis.
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dr = float(pts[i, 0] - pts[j, 0])
            dc = float(pts[i, 1] - pts[j, 1])
            d2 = dr * dr + dc * dc
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j

    if best_d2 <= 1e-12:
        return None
    p0 = pts[best_i, ::-1].astype(np.float64)
    p1 = pts[best_j, ::-1].astype(np.float64)
    return np.vstack((p0, p1))


@njit(nogil=True, cache=True)
def _equivalent_circle_radius(mask):
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return 0.0
    return float(np.sqrt(area / np.pi))


@njit(nogil=True, cache=True)
def _fallback_round_cell(mask, dt):
    """Round-cell mesh for 0/1-point skeletons.

    The spine is represented by three coincident centers and one vertical rib.
    Thus spine_length=0, length=2*r, width=2*r, area=pi*r^2, volume=4/3*pi*r^3.
    """
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None, None
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_dt = float(dt[int(round(cy)), int(round(cx))])
    r_area = _equivalent_circle_radius(mask)
    r = float(np.median(np.array((r_dt, r_area))))
    if r < 0.5:
        return None, None

    mesh = np.zeros((3, 4))
    # Vertical rib through the center. Duplicate rows keep the mesh API
    # compatible while contributing zero body spine length.
    for i in range(3):
        mesh[i, 0] = cx
        mesh[i, 1] = cy - r
        mesh[i, 2] = cx
        mesh[i, 3] = cy + r
    return mesh, (r, r)


@njit(nogil=True, cache=True)
def _fallback_two_point_cell(cl2, mask, dt):
    """Short-capsule mesh for cells with exactly two valid medial points."""
    if cl2 is None or len(cl2) != 2:
        return None, None

    p0 = np.asarray(cl2[0], dtype=np.float64)
    p1 = np.asarray(cl2[1], dtype=np.float64)
    axis = p1 - p0
    spine_len = float(np.sqrt(np.sum(axis * axis)))
    if spine_len < 1e-8:
        return _fallback_round_cell(mask, dt)
    axis /= spine_len
    normal = np.array((-axis[1], axis[0]), dtype=np.float64)

    radii_dt = _sample_dt(cl2.astype(np.float64), dt)
    r_dt = float(np.median(radii_dt))
    area = float(np.count_nonzero(mask))
    # Area-preserving capsule radius from A = 2*r*L + pi*r^2.
    r_area = float((-spine_len + np.sqrt(spine_len * spine_len + np.pi * area)) / np.pi)
    r = float(np.median(np.array((r_dt, r_area))))
    if not np.isfinite(r) or r < 0.5:
        return None, None

    centers = np.vstack((p0, (p0 + p1) / 2.0, p1))
    mesh = np.zeros((3, 4))
    for i in range(3):
        c = centers[i]
        mesh[i, 0] = c[0] + normal[0] * r
        mesh[i, 1] = c[1] + normal[1] * r
        mesh[i, 2] = c[0] - normal[0] * r
        mesh[i, 3] = c[1] - normal[1] * r
    return mesh, (r, r)


@njit(nogil=True, cache=True)
def _order_points(pts_rc):
    """Order skeleton points along the main endpoint path of an 8-connected graph."""
    n = len(pts_rc)
    if n < 2:
        return None

    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            dr = float(pts_rc[i, 0]) - float(pts_rc[j, 0])
            dc = float(pts_rc[i, 1]) - float(pts_rc[j, 1])
            if dr*dr + dc*dc <= 2.01:
                degree[i] += 1
                degree[j] += 1
    endpoints = np.where(degree == 1)[0]
    if len(endpoints) >= 2 and np.any(degree > 2):
        # A greedy walk can enter a short pole spur and abandon the cell body.
        # Choose the longest endpoint-to-endpoint geodesic instead. Shortest
        # paths avoid loops caused by diagonal links at skeleton junctions.
        weights = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dr = float(pts_rc[i, 0]) - float(pts_rc[j, 0])
                dc = float(pts_rc[i, 1]) - float(pts_rc[j, 1])
                d2 = dr * dr + dc * dc
                if 0 < d2 <= 2.01:
                    weights[i, j] = weights[j, i] = np.sqrt(d2)

        best_length = -1.0
        best_path = np.empty(n, dtype=np.int64)
        best_count = 0
        for source_idx in range(len(endpoints) - 1):
            source = endpoints[source_idx]
            distances = np.full(n, np.inf)
            previous = np.full(n, -1, dtype=np.int64)
            settled = np.zeros(n, dtype=np.bool_)
            distances[source] = 0.0
            for _ in range(n):
                current = -1
                nearest = np.inf
                for i in range(n):
                    if not settled[i] and distances[i] < nearest:
                        current, nearest = i, distances[i]
                if current < 0:
                    break
                settled[current] = True
                for nb in range(n):
                    candidate = nearest + weights[current, nb]
                    if not settled[nb] and candidate < distances[nb]:
                        distances[nb] = candidate
                        previous[nb] = current
            for target_idx in range(source_idx + 1, len(endpoints)):
                target = endpoints[target_idx]
                length = distances[target]
                if not np.isfinite(length) or length <= best_length:
                    continue
                path = np.empty(n, dtype=np.int64)
                count = 0
                current = target
                while current >= 0 and count < n:
                    path[count] = current
                    count += 1
                    if current == source:
                        break
                    current = previous[current]
                if current == source and count >= 3:
                    best_length = length
                    best_count = count
                    for i in range(count):
                        best_path[i] = path[count - 1 - i]
        if best_count >= 3:
            return pts_rc[best_path[:best_count]]

    start = 0
    for i in range(n):
        if degree[i] == 1:
            start = i
            break

    visited = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    order[0] = start
    order_count = 1
    visited[start] = True
    current = start
    for _ in range(n):
        nxt = -1
        for nb in range(n):
            if visited[nb]:
                continue
            dr = float(pts_rc[current, 0]) - float(pts_rc[nb, 0])
            dc = float(pts_rc[current, 1]) - float(pts_rc[nb, 1])
            if dr*dr + dc*dc <= 2.01:
                nxt = nb
                break
        if nxt < 0:
            break
        visited[nxt] = True
        order[order_count] = nxt
        order_count += 1
        current = nxt
    if order_count < 3:
        return None
    return pts_rc[order[:order_count]]


@njit(nogil=True, cache=True)
def _moving_average(pts, window):
    """Moving average for 2D polyline."""
    if len(pts) < window or window < 2:
        return pts
    result = pts.copy()
    half = window // 2
    for i in range(len(pts)):
        lo, hi = max(0, i - half), min(len(pts), i + half + 1)
        result[i, 0] = np.mean(pts[lo:hi, 0])
        result[i, 1] = np.mean(pts[lo:hi, 1])
    return result


@njit(nogil=True, cache=True)
def _moving_average_1d(vals, window):
    """Moving average for 1D array."""
    if len(vals) < window or window < 2:
        return vals
    result = vals.copy()
    half = window // 2
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        result[i] = np.mean(vals[lo:hi])
    return result


def _spline_smooth(pts, tol):
    """Cubic smoothing spline. tol > 0 controls smoothing amount."""
    d = np.diff(pts, axis=0)
    l = np.cumsum(np.hstack([[0], np.sqrt((d * d).sum(axis=1))]))
    # Use spatial extent, not a flattened x/y range: the latter changes when
    # the same cell is translated or cropped within an image.
    ptp_val = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
    s_value = max(len(pts) * 0.1, tol * len(pts) * ptp_val * 0.05)
    origin = pts[0].copy()
    centered = pts - origin
    try:
        tck_x = splrep(l, centered[:, 0], s=s_value, k=min(3, len(pts) - 1))
        tck_y = splrep(l, centered[:, 1], s=s_value, k=min(3, len(pts) - 1))
        result = np.column_stack([splev(l, tck_x), splev(l, tck_y)]) + origin
        if np.any(np.isnan(result)):
            return pts
        return result
    except Exception:
        return pts


@njit(nogil=True, cache=True)
def _interp_equal_steps(pts, step):
    """Interpolate polyline to equal arc-length steps."""
    if len(pts) < 2:
        # cast for dtype-consistent returns: numba must unify this branch
        # with the float64 column_stack below even when pts is int64
        return pts.astype(np.float64)
    l = np.zeros(len(pts), dtype=np.float64)
    for i in range(1, len(pts)):
        dx = pts[i, 0] - pts[i - 1, 0]
        dy = pts[i, 1] - pts[i - 1, 1]
        l[i] = l[i - 1] + np.sqrt(dx * dx + dy * dy)
    total = float(l[-1])
    if np.isnan(total) or total <= 0 or step <= 0 or total < step:
        return pts.astype(np.float64)
    n_steps = max(3, int(total / step) + 1)
    ev = np.linspace(0, total, n_steps)
    return np.column_stack((np.interp(ev, l, pts[:, 0]),
                            np.interp(ev, l, pts[:, 1])))


@njit(nogil=True, cache=True)
def _sample_dt(cl, dt):
    """Bilinear sample of distance transform at sub-pixel centerline pts."""
    h, w = dt.shape
    radii = np.zeros(len(cl))
    for i in range(len(cl)):
        x, y = cl[i, 0], cl[i, 1]
        x0 = max(0, min(w - 2, int(np.floor(x))))
        y0 = max(0, min(h - 2, int(np.floor(y))))
        x1, y1 = x0 + 1, y0 + 1
        wx, wy = x - x0, y - y0
        radii[i] = (dt[y0, x0] * (1 - wx) * (1 - wy) +
                    dt[y0, x1] * wx * (1 - wy) +
                    dt[y1, x0] * (1 - wx) * wy +
                    dt[y1, x1] * wx * wy)
    return radii


# =============================================================================
# 3. Rib generation
# =============================================================================

@njit(nogil=True, cache=True)
def _build_ribs(cl, radii):
    """Build ribs: left = cl + normal*radius, right = cl - normal*radius."""
    n = len(cl)
    tx = np.zeros(n)
    ty = np.zeros(n)
    for i in range(1, n - 1):
        tx[i] = cl[i + 1, 0] - cl[i - 1, 0]
        ty[i] = cl[i + 1, 1] - cl[i - 1, 1]
    tx[0] = cl[1, 0] - cl[0, 0]
    ty[0] = cl[1, 1] - cl[0, 1]
    tx[-1] = cl[-1, 0] - cl[-2, 0]
    ty[-1] = cl[-1, 1] - cl[-2, 1]

    nrm = np.sqrt(tx*tx + ty*ty)
    for i in range(n):
        if nrm[i] < 1e-14:
            nrm[i] = 1.0
    tx_s, ty_s = tx / nrm, ty / nrm
    if len(tx_s) >= 5:
        w = max(3, len(tx_s) // 6 * 2 + 1)
        tx_s = _moving_average_1d(tx_s, w)
        ty_s = _moving_average_1d(ty_s, w)
        nrm2 = np.sqrt(tx_s*tx_s + ty_s*ty_s)
        for i in range(n):
            if nrm2[i] < 1e-14:
                nrm2[i] = 1.0
        tx_s /= nrm2
        ty_s /= nrm2

    nx, ny = -ty_s, tx_s
    mesh = np.zeros((n, 4))
    for i in range(n):
        r = radii[i]
        mesh[i, 0] = cl[i, 0] + nx[i] * r
        mesh[i, 1] = cl[i, 1] + ny[i] * r
        mesh[i, 2] = cl[i, 0] - nx[i] * r
        mesh[i, 3] = cl[i, 1] - ny[i] * r
    return mesh


@njit(nogil=True, cache=True)
def _close_contour(contour_xy):
    """Return a closed contour in [x, y] order."""
    if contour_xy is None or len(contour_xy) < 3:
        return None
    contour_xy = np.asarray(contour_xy, dtype=np.float64)
    if np.any(~np.isfinite(contour_xy)):
        return None
    if np.linalg.norm(contour_xy[0] - contour_xy[-1]) > 1e-8:
        closed = np.empty((len(contour_xy) + 1, 2), dtype=np.float64)
        for i in range(len(contour_xy)):
            closed[i, 0] = contour_xy[i, 0]
            closed[i, 1] = contour_xy[i, 1]
        closed[-1, 0] = contour_xy[0, 0]
        closed[-1, 1] = contour_xy[0, 1]
        contour_xy = closed
    return contour_xy


def _centerline_normals(cl):
    """Compute smoothed unit normals for a centerline."""
    n = len(cl)
    if n < 2:
        return None
    tx = np.zeros(n)
    ty = np.zeros(n)
    for i in range(1, n - 1):
        tx[i] = cl[i + 1, 0] - cl[i - 1, 0]
        ty[i] = cl[i + 1, 1] - cl[i - 1, 1]
    tx[0] = cl[1, 0] - cl[0, 0]
    ty[0] = cl[1, 1] - cl[0, 1]
    tx[-1] = cl[-1, 0] - cl[-2, 0]
    ty[-1] = cl[-1, 1] - cl[-2, 1]

    nrm = np.sqrt(tx * tx + ty * ty)
    nrm[nrm < 1e-14] = 1.0
    tx_s, ty_s = tx / nrm, ty / nrm
    if n >= 5:
        w = max(3, n // 6 * 2 + 1)
        tx_s = _moving_average_1d(tx_s, w)
        ty_s = _moving_average_1d(ty_s, w)
        nrm2 = np.sqrt(tx_s * tx_s + ty_s * ty_s)
        nrm2[nrm2 < 1e-14] = 1.0
        tx_s /= nrm2
        ty_s /= nrm2
    return np.column_stack([-ty_s, tx_s])


@njit(nogil=True, cache=True)
def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _line_contour_intersections(point, normal, contour_xy):
    """Find intersections of p + t*n with a closed contour."""
    hits = []
    if contour_xy is None or len(contour_xy) < 4:
        return hits
    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return hits
    n = n / nn

    for i in range(len(contour_xy) - 1):
        a = contour_xy[i]
        b = contour_xy[i + 1]
        s = b - a
        den = _cross2(n, s)
        if abs(den) < 1e-12:
            continue
        ap = a - p
        t = _cross2(ap, s) / den
        u = _cross2(ap, n) / den
        if -1e-9 <= u <= 1.0 + 1e-9:
            u_clip = min(1.0, max(0.0, u))
            pos = float(i + u_clip)
            pt = a + u_clip * s
            if not np.all(np.isfinite(pt)):
                continue
            duplicate = False
            for old_t, _, old_pos in hits:
                if abs(old_t - t) < 1e-6 or abs(old_pos - pos) < 1e-6:
                    duplicate = True
                    break
            if not duplicate:
                hits.append((float(t), pt, pos))
    return hits


def _candidate_score(t, expected_radius):
    expected = max(float(expected_radius), 0.5)
    return abs(abs(t) - expected) / (expected + 0.5)


def _select_monotonic_side(candidates_by_rib, period):
    """Choose one contour intersection per rib with monotonic contour order."""
    if any(len(cands) == 0 for cands in candidates_by_rib):
        return None
    best_seq, best_cost = None, np.inf
    first_candidates = sorted(candidates_by_rib[0], key=lambda c: c["score"])[:3]
    for direction in (1.0, -1.0):
        for first in first_candidates:
            seq = [first]
            prev_progress = direction * first["pos"]
            cost = first["score"]
            ok = True
            for rib_cands in candidates_by_rib[1:]:
                best = None
                best_local_cost = np.inf
                best_progress = None
                for cand in rib_cands:
                    progress = direction * cand["pos"]
                    while progress < prev_progress - 0.25:
                        progress += period
                    jump = max(0.0, progress - prev_progress)
                    local_cost = cand["score"] + 0.01 * jump
                    if local_cost < best_local_cost:
                        best = cand
                        best_local_cost = local_cost
                        best_progress = progress
                if best is None:
                    ok = False
                    break
                seq.append(best)
                prev_progress = best_progress
                cost += best_local_cost
            if ok and cost < best_cost:
                best_seq = seq
                best_cost = cost
    return best_seq


@njit(nogil=True, cache=True)
def _seed_endpoints_have_clearance(cl, contour_xy):
    """Check the two seed endpoints using the rib builder's signed-hit cutoff."""
    normals, ok = _centerline_normals_numba(cl)
    if not ok or len(contour_xy) < 4:
        return False
    for index in (0, len(cl) - 1):
        point = cl[index]
        if not _point_in_polygon(point, contour_xy):
            return False
        normal = normals[index]
        norm = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm < 1e-12:
            return False
        normal = normal / norm
        has_positive, has_negative = False, False
        for edge_index in range(len(contour_xy) - 1):
            start = contour_xy[edge_index]
            segment = contour_xy[edge_index + 1] - start
            den = _cross2(normal, segment)
            if abs(den) < 1e-12:
                continue
            offset = start - point
            t = _cross2(offset, segment) / den
            u = _cross2(offset, normal) / den
            if -1e-9 <= u <= 1.0 + 1e-9:
                if t >= 0.25:
                    has_positive = True
                if t <= -0.25:
                    has_negative = True
        if not has_positive or not has_negative:
            return False
    return True


def _guard_spline_seed_endpoints(cl, supported_cl, dt, contour_xy):
    """Use the pre-spline seed only when endpoint clearance blocks contour ribs.

    The spline is unconstrained and can move an endpoint outside a narrow pole.
    Retrying refinement cannot help when the very first rib cannot be built.
    Preserve the original seed bit-for-bit when its endpoints are supported;
    any replacement must itself produce a strictly valid contour-rib mesh.
    """
    if contour_xy is None or _seed_endpoints_have_clearance(cl, contour_xy):
        return cl
    radii = _sample_refinement_radii(supported_cl, dt)
    candidate = _build_ribs_contour_constrained(supported_cl, radii, contour_xy)
    if candidate is not None and _validate_non_crossing_mesh(candidate, radii):
        return supported_cl
    return cl


def _build_ribs_contour_constrained(cl, radii, contour_xy):
    """Build ribs from contour intersections instead of free DT extension."""
    if contour_xy is None or len(cl) < 3 or len(cl) != len(radii):
        return None
    mesh, ok = _build_ribs_contour_constrained_numba(
        np.asarray(cl, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        np.asarray(contour_xy, dtype=np.float64),
    )
    if not ok:
        return None
    return mesh


@njit(nogil=True, cache=True)
def _centerline_normals_numba(cl):
    """Compute smoothed unit normals for a centerline."""
    n = len(cl)
    normals = np.zeros((n, 2))
    if n < 2:
        return normals, False

    tx = np.zeros(n)
    ty = np.zeros(n)
    for i in range(1, n - 1):
        tx[i] = cl[i + 1, 0] - cl[i - 1, 0]
        ty[i] = cl[i + 1, 1] - cl[i - 1, 1]
    tx[0] = cl[1, 0] - cl[0, 0]
    ty[0] = cl[1, 1] - cl[0, 1]
    tx[n - 1] = cl[n - 1, 0] - cl[n - 2, 0]
    ty[n - 1] = cl[n - 1, 1] - cl[n - 2, 1]

    for i in range(n):
        nrm = np.sqrt(tx[i] * tx[i] + ty[i] * ty[i])
        if nrm < 1e-14:
            nrm = 1.0
        tx[i] /= nrm
        ty[i] /= nrm

    if n >= 5:
        w = max(3, n // 6 * 2 + 1)
        tx = _moving_average_1d(tx, w)
        ty = _moving_average_1d(ty, w)
        for i in range(n):
            nrm = np.sqrt(tx[i] * tx[i] + ty[i] * ty[i])
            if nrm < 1e-14:
                nrm = 1.0
            tx[i] /= nrm
            ty[i] /= nrm

    for i in range(n):
        normals[i, 0] = -ty[i]
        normals[i, 1] = tx[i]
    return normals, True


@njit(nogil=True, cache=True)
def _candidate_score_numba(t, expected_radius):
    expected = max(float(expected_radius), 0.5)
    return abs(abs(t) - expected) / (expected + 0.5)


@njit(nogil=True, cache=True)
def _insert_sorted_candidate(xs, ys, poss, scores, count, x, y, pos, score):
    max_candidates = xs.shape[0]
    if count < max_candidates:
        idx = count
        count += 1
    elif score < scores[max_candidates - 1]:
        idx = max_candidates - 1
    else:
        return count

    xs[idx] = x
    ys[idx] = y
    poss[idx] = pos
    scores[idx] = score
    while idx > 0 and scores[idx] < scores[idx - 1]:
        tx = xs[idx - 1]
        ty = ys[idx - 1]
        tp = poss[idx - 1]
        ts = scores[idx - 1]
        xs[idx - 1] = xs[idx]
        ys[idx - 1] = ys[idx]
        poss[idx - 1] = poss[idx]
        scores[idx - 1] = scores[idx]
        xs[idx] = tx
        ys[idx] = ty
        poss[idx] = tp
        scores[idx] = ts
        idx -= 1
    return count


@njit(nogil=True, cache=True)
def _select_side_points_numba(xs, ys, poss, scores, counts, period):
    n_ribs = xs.shape[0]
    selected = np.zeros((n_ribs, 2))
    best_selected = np.zeros((n_ribs, 2))
    best_cost = np.inf

    for i in range(n_ribs):
        if counts[i] == 0:
            return best_selected, False

    first_count = min(3, counts[0])
    for direction_idx in range(2):
        direction = 1.0
        if direction_idx == 1:
            direction = -1.0
        for first_idx in range(first_count):
            selected[0, 0] = xs[0, first_idx]
            selected[0, 1] = ys[0, first_idx]
            prev_progress = direction * poss[0, first_idx]
            cost = scores[0, first_idx]
            ok = True

            for rib_idx in range(1, n_ribs):
                best_local_cost = np.inf
                best_progress = 0.0
                best_cand = -1
                for cand_idx in range(counts[rib_idx]):
                    progress = direction * poss[rib_idx, cand_idx]
                    while progress < prev_progress - 0.25:
                        progress += period
                    jump = max(0.0, progress - prev_progress)
                    local_cost = scores[rib_idx, cand_idx] + 0.01 * jump
                    if local_cost < best_local_cost:
                        best_local_cost = local_cost
                        best_progress = progress
                        best_cand = cand_idx
                if best_cand < 0:
                    ok = False
                    break
                selected[rib_idx, 0] = xs[rib_idx, best_cand]
                selected[rib_idx, 1] = ys[rib_idx, best_cand]
                prev_progress = best_progress
                cost += best_local_cost

            if ok and cost < best_cost:
                best_cost = cost
                for rib_idx in range(n_ribs):
                    best_selected[rib_idx, 0] = selected[rib_idx, 0]
                    best_selected[rib_idx, 1] = selected[rib_idx, 1]

    return best_selected, np.isfinite(best_cost)


@njit(nogil=True, cache=True)
def _build_ribs_contour_constrained_numba(cl, radii, contour_xy):
    n_ribs = len(cl)
    mesh = np.zeros((n_ribs, 4))
    if len(contour_xy) < 4 or n_ribs < 3 or n_ribs != len(radii):
        return mesh, False

    normals, ok = _centerline_normals_numba(cl)
    if not ok:
        return mesh, False

    max_candidates = 6
    pos_x = np.zeros((n_ribs, max_candidates))
    pos_y = np.zeros((n_ribs, max_candidates))
    pos_pos = np.zeros((n_ribs, max_candidates))
    pos_score = np.empty((n_ribs, max_candidates))
    neg_x = np.zeros((n_ribs, max_candidates))
    neg_y = np.zeros((n_ribs, max_candidates))
    neg_pos = np.zeros((n_ribs, max_candidates))
    neg_score = np.empty((n_ribs, max_candidates))
    pos_count = np.zeros(n_ribs, dtype=np.int64)
    neg_count = np.zeros(n_ribs, dtype=np.int64)
    for i in range(n_ribs):
        for j in range(max_candidates):
            pos_score[i, j] = np.inf
            neg_score[i, j] = np.inf

    n_edges = len(contour_xy) - 1
    hit_t = np.zeros(n_edges)
    hit_pos = np.zeros(n_edges)

    for rib_idx in range(n_ribs):
        px = cl[rib_idx, 0]
        py = cl[rib_idx, 1]
        nx = normals[rib_idx, 0]
        ny = normals[rib_idx, 1]
        nn = np.sqrt(nx * nx + ny * ny)
        if nn < 1e-12:
            continue
        nx /= nn
        ny /= nn
        n_hits = 0

        for edge_idx in range(n_edges):
            ax = contour_xy[edge_idx, 0]
            ay = contour_xy[edge_idx, 1]
            sx = contour_xy[edge_idx + 1, 0] - ax
            sy = contour_xy[edge_idx + 1, 1] - ay
            den = nx * sy - ny * sx
            if abs(den) < 1e-12:
                continue

            apx = ax - px
            apy = ay - py
            t = (apx * sy - apy * sx) / den
            u = (apx * ny - apy * nx) / den
            if -1e-9 <= u <= 1.0 + 1e-9:
                u_clip = min(1.0, max(0.0, u))
                contour_pos = float(edge_idx) + u_clip
                duplicate = False
                for hit_idx in range(n_hits):
                    if (abs(hit_t[hit_idx] - t) < 1e-6 or
                            abs(hit_pos[hit_idx] - contour_pos) < 1e-6):
                        duplicate = True
                        break
                if duplicate:
                    continue
                if n_hits < n_edges:
                    hit_t[n_hits] = t
                    hit_pos[n_hits] = contour_pos
                    n_hits += 1

                if abs(t) < 0.25:
                    continue
                hx = ax + u_clip * sx
                hy = ay + u_clip * sy
                if not (np.isfinite(hx) and np.isfinite(hy)):
                    continue
                score = _candidate_score_numba(t, radii[rib_idx])
                if t > 0:
                    pos_count[rib_idx] = _insert_sorted_candidate(
                        pos_x[rib_idx], pos_y[rib_idx], pos_pos[rib_idx],
                        pos_score[rib_idx], pos_count[rib_idx],
                        hx, hy, contour_pos, score)
                else:
                    neg_count[rib_idx] = _insert_sorted_candidate(
                        neg_x[rib_idx], neg_y[rib_idx], neg_pos[rib_idx],
                        neg_score[rib_idx], neg_count[rib_idx],
                        hx, hy, contour_pos, score)

    period = max(float(n_edges), 1.0)
    pos_side, pos_ok = _select_side_points_numba(
        pos_x, pos_y, pos_pos, pos_score, pos_count, period)
    if not pos_ok:
        return mesh, False
    neg_side, neg_ok = _select_side_points_numba(
        neg_x, neg_y, neg_pos, neg_score, neg_count, period)
    if not neg_ok:
        return mesh, False

    for i in range(n_ribs):
        mesh[i, 0] = pos_side[i, 0]
        mesh[i, 1] = pos_side[i, 1]
        mesh[i, 2] = neg_side[i, 0]
        mesh[i, 3] = neg_side[i, 1]
    return mesh, True


@njit(nogil=True, cache=True)
def _interp_n_points(pts, n):
    """Interpolate a polyline to exactly n equal-arc positions."""
    if pts is None or len(pts) < 2 or n < 2:
        # cast for dtype-consistent returns (int64 inputs must not leak out)
        return pts.astype(np.float64)
    l = np.zeros(len(pts), dtype=np.float64)
    for i in range(1, len(pts)):
        dx = pts[i, 0] - pts[i - 1, 0]
        dy = pts[i, 1] - pts[i - 1, 1]
        l[i] = l[i - 1] + np.sqrt(dx * dx + dy * dy)
    total = float(l[-1])
    if not np.isfinite(total) or total <= 0:
        return pts.astype(np.float64)
    ev = np.linspace(0, total, n)
    return np.column_stack((np.interp(ev, l, pts[:, 0]),
                            np.interp(ev, l, pts[:, 1])))


@njit(nogil=True, cache=True)
def _refinement_smooth_window(n_points, smooth_tol):
    if n_points < 5:
        return 1
    window = max(3, min(7, n_points // 5 * 2 + 1))
    if smooth_tol is not None and smooth_tol >= 0.05:
        window = min(11, window + 2)
        if window % 2 == 0:
            window += 1
    return window


@njit(nogil=True, cache=True)
def _clip_spine_displacement(displacement, radii, step_size):
    """Clip noisy rib-midpoint jumps before they move the spine."""
    disp = displacement.copy()
    n_points = len(disp)
    mag = np.empty(n_points, dtype=np.float64)
    finite_count = 0
    for i in range(n_points):
        mag[i] = np.sqrt(disp[i, 0] * disp[i, 0] +
                         disp[i, 1] * disp[i, 1])
        if np.isfinite(mag[i]):
            finite_count += 1
    if finite_count == 0:
        return disp * 0.0

    if radii is not None:
        valid_radii = np.empty(len(radii), dtype=np.float64)
        valid_radii_count = 0
        for i in range(len(radii)):
            if np.isfinite(radii[i]) and radii[i] > 0:
                valid_radii[valid_radii_count] = radii[i]
                valid_radii_count += 1
        if valid_radii_count:
            radius_limit = float(
                np.median(valid_radii[:valid_radii_count]))
        else:
            radius_limit = 0.0
    else:
        radius_limit = 0.0
    max_update = max(0.75, min(2.5, max(radius_limit, 1.5 * step_size)))

    finite_mag = np.empty(finite_count, dtype=np.float64)
    out_i = 0
    for i in range(n_points):
        if np.isfinite(mag[i]):
            finite_mag[out_i] = mag[i]
            out_i += 1
    med = float(np.median(finite_mag))
    deviations = np.empty(finite_count, dtype=np.float64)
    for i in range(finite_count):
        deviations[i] = abs(finite_mag[i] - med)
    mad = float(np.median(deviations))
    robust_limit = med + 3.0 * max(1e-6, 1.4826 * mad) + 0.25
    limit = max(0.5, min(max_update, robust_limit))

    for i in range(n_points):
        scale = 1.0
        if not np.isfinite(mag[i]):
            scale = 0.0
        elif mag[i] > limit:
            scale = limit / (mag[i] + 1e-12)
        disp[i, 0] *= scale
        disp[i, 1] *= scale
    return disp


@njit(nogil=True, cache=True)
def _damped_centerline_update(current_cl, observed_cl, step_size, smooth_tol,
                              radii=None, fixed_endpoints=False,
                              update_beta=0.45):
    """Move a spine partway toward noisy rib midpoints, then smooth it."""
    if (current_cl is None or observed_cl is None or len(current_cl) < 3 or
            len(observed_cl) < 3):
        return None
    current = np.asarray(current_cl, dtype=np.float64)
    observed = _interp_n_points(
        np.asarray(observed_cl, dtype=np.float64), len(current))
    if observed is None or len(observed) != len(current):
        return None
    if np.any(~np.isfinite(current)) or np.any(~np.isfinite(observed)):
        return None

    first = current[0].copy()
    last = current[-1].copy()
    displacement = _clip_spine_displacement(observed - current, radii, step_size)
    if fixed_endpoints:
        displacement[0] = 0.0
        displacement[-1] = 0.0

    window = _refinement_smooth_window(len(displacement), smooth_tol)
    if window >= 3:
        displacement = _moving_average(displacement, window)
    if fixed_endpoints:
        displacement[0] = 0.0
        displacement[-1] = 0.0

    beta = max(0.05, min(1.0, float(update_beta)))
    updated = current + beta * displacement
    if fixed_endpoints:
        updated[0] = first
        updated[-1] = last

    if fixed_endpoints:
        return _smooth_refinement_centerline_fixed_endpoints(
            updated, step_size, smooth_tol)
    return _smooth_refinement_centerline(updated, step_size, smooth_tol)


@njit(nogil=True, cache=True)
def _smooth_refinement_centerline(cl, step_size, smooth_tol):
    """Smooth and resample a centerline during fixed-point refinement."""
    if cl is None or len(cl) < 3:
        return None
    out = cl.copy()
    w = _refinement_smooth_window(len(out), smooth_tol)
    if w >= 3:
        out = _moving_average(out, w)
    out = _interp_equal_steps(out, step_size)
    if out is None or len(out) < 3 or np.any(~np.isfinite(out)):
        return None
    return out


@njit(nogil=True, cache=True)
def _smooth_refinement_centerline_fixed_endpoints(cl, step_size, smooth_tol):
    """Smooth and resample a centerline while preserving pole centers."""
    if cl is None or len(cl) < 3:
        return None
    first = cl[0].copy()
    last = cl[-1].copy()
    out = cl.copy()
    w = _refinement_smooth_window(len(out), smooth_tol)
    if w >= 3:
        out = _moving_average(out, w)
        out[0] = first
        out[-1] = last
    out = _interp_equal_steps(out, step_size)
    if out is None or len(out) < 3 or np.any(~np.isfinite(out)):
        return None
    out[0] = first
    out[-1] = last
    if np.linalg.norm(out[1] - out[0]) < 0.1:
        out = out[1:]
        out[0] = first
    if len(out) < 3 or np.linalg.norm(out[-1] - out[-2]) < 0.1:
        out = out[:-1]
        out[-1] = last
    if len(out) < 3 or np.any(~np.isfinite(out)):
        return None
    return out


@njit(nogil=True, cache=True)
def _mean_centerline_shift(old_cl, new_cl):
    if old_cl is None or new_cl is None or len(old_cl) < 2 or len(new_cl) < 2:
        return np.inf
    new_match = _interp_n_points(new_cl, len(old_cl))
    if new_match is None or len(new_match) != len(old_cl):
        return np.inf
    total = 0.0
    for i in range(len(old_cl)):
        dx = old_cl[i, 0] - new_match[i, 0]
        dy = old_cl[i, 1] - new_match[i, 1]
        total += np.sqrt(dx * dx + dy * dy)
    return float(total / len(old_cl))


@njit(nogil=True, cache=True)
def _mesh_centers(mesh):
    centers = np.empty((len(mesh), 2), dtype=np.float64)
    for i in range(len(mesh)):
        centers[i, 0] = (mesh[i, 0] + mesh[i, 2]) / 2.0
        centers[i, 1] = (mesh[i, 1] + mesh[i, 3]) / 2.0
    return centers


@njit(nogil=True, cache=True)
def _sample_refinement_radii(cl, dt):
    radii = _sample_dt(cl, dt)
    if len(radii) >= 3:
        radii = _moving_average_1d(radii, max(3, len(radii) // 8 * 2 + 1))
    return radii


def _refine_mesh_fixed_point(cl, dt, contour_xy, step_size, smooth_tol,
                             max_refine_iter=5, spine_update_beta=0.45):
    """Iteratively rebuild contour ribs and recenter the centerline."""
    current_cl = cl.copy()
    best_mesh = None
    best_radii = None
    best_shift = np.inf

    for _ in range(max_refine_iter):
        radii = _sample_refinement_radii(current_cl, dt)
        mesh = _build_ribs_contour_constrained(current_cl, radii, contour_xy)
        if mesh is None:
            break

        centers = _mesh_centers(mesh)
        shift = _mean_centerline_shift(current_cl, centers)
        valid = _validate_non_crossing_mesh(mesh, radii)
        if valid and shift < best_shift:
            best_mesh = mesh
            best_radii = radii
            best_shift = shift
        if valid and shift < 0.25:
            return mesh, radii

        next_cl = _damped_centerline_update(
            current_cl, centers, step_size, smooth_tol, radii=radii,
            update_beta=spine_update_beta)
        if next_cl is None:
            break
        current_cl = next_cl

    if best_mesh is not None:
        return best_mesh, best_radii
    return None


@njit(nogil=True, cache=True)
def _pole_centers_from_mesh(mesh, contour_xy, r_pole):
    """Find pole-tip-derived cap centers for the current body mesh."""
    if (mesh is None or contour_xy is None or len(mesh) < 3 or
            not np.isfinite(r_pole) or r_pole <= 0):
        return None
    centers = _mesh_centers(mesh)
    first_dir = _unit_vector(centers[0] - centers[1])
    last_dir = _unit_vector(centers[-1] - centers[-2])
    if first_dir is None or last_dir is None:
        return None

    first_tip = _find_pole_tip(contour_xy, centers[0], first_dir, r_pole)
    last_tip = _find_pole_tip(contour_xy, centers[-1], last_dir, r_pole)
    if first_tip is None or last_tip is None:
        return None

    first_center = first_tip - first_dir * r_pole
    last_center = last_tip - last_dir * r_pole
    if (not _point_in_polygon(first_center, contour_xy) or
            not _point_in_polygon(last_center, contour_xy)):
        return None
    return first_center, last_center, first_dir, last_dir


def _centerline_with_integrated_poles(centers, pole_info, step_size, r_pole):
    """Insert cap centers and discard body midpoints that lie inside caps."""
    first_center, last_center, first_dir, last_dir = pole_info
    min_gap = max(0.25, min(0.5 * float(step_size), 0.25 * float(r_pole)))
    body = []
    for p in centers:
        from_first = float(np.dot(p - first_center, first_dir))
        from_last = float(np.dot(p - last_center, last_dir))
        if from_first <= -min_gap and from_last <= -min_gap:
            body.append(p)
    if len(body) == 0:
        return None
    cl = np.vstack([first_center.reshape(1, 2),
                    np.asarray(body, dtype=float),
                    last_center.reshape(1, 2)])
    keep = [cl[0]]
    min_dist = max(0.1, min_gap * 0.25)
    for p in cl[1:]:
        if np.linalg.norm(p - keep[-1]) >= min_dist:
            keep.append(p)
    if len(keep) < 2 or np.linalg.norm(keep[-1] - last_center) >= min_dist:
        keep.append(last_center)
    cl = np.asarray(keep, dtype=float)
    if len(cl) < 3 or np.any(~np.isfinite(cl)):
        return None
    return cl


def _build_integrated_pole_mesh(cl, dt, contour_xy, pole_info, r_pole):
    """Build a contour-rib mesh whose first and last ribs are pole caps."""
    radii = _sample_refinement_radii(cl, dt)
    if len(radii) != len(cl):
        return None
    radii[0] = r_pole
    radii[-1] = r_pole
    mesh = _build_ribs_contour_constrained(cl, radii, contour_xy)
    if mesh is None:
        return None
    first_center, last_center, _, _ = pole_info
    if not _set_endpoint_rib(mesh, 0, first_center, cl[1] - first_center, r_pole):
        return None
    if not _set_endpoint_rib(mesh, -1, last_center, last_center - cl[-2], r_pole):
        return None
    return _trim_endpoint_crossing_ribs(mesh)


def _refine_mesh_with_integrated_poles(cl, dt, contour_xy, pole_contour_xy,
                                       step_size, smooth_tol,
                                       max_refine_iter=5,
                                       spine_update_beta=0.45):
    """Fixed-point refinement with pole centers as endpoint constraints."""
    if pole_contour_xy is None:
        pole_contour_xy = contour_xy
    current_cl = cl.copy()
    best_mesh = None
    best_r_pole = None
    best_shift = np.inf

    for _ in range(max_refine_iter):
        radii = _sample_refinement_radii(current_cl, dt)
        body_mesh = _build_ribs_contour_constrained(current_cl, radii, contour_xy)
        if body_mesh is None:
            break

        centers = _mesh_centers(body_mesh)
        r_pole = _estimate_pole_radius_from_mesh(body_mesh)
        if not np.isfinite(r_pole) or r_pole <= 0:
            r_pole = float(np.median(radii))
        pole_info = _pole_centers_from_mesh(body_mesh, pole_contour_xy, r_pole)
        if pole_info is None:
            next_cl = _damped_centerline_update(
                current_cl, centers, step_size, smooth_tol, radii=radii,
                update_beta=spine_update_beta)
            if next_cl is None:
                break
            current_cl = next_cl
            continue

        observed_pole_cl = _centerline_with_integrated_poles(
            centers, pole_info, step_size, r_pole)
        if observed_pole_cl is None:
            break
        current_pole_cl = _centerline_with_integrated_poles(
            current_cl, pole_info, step_size, r_pole)
        if current_pole_cl is None:
            current_pole_cl = observed_pole_cl
        pole_cl = _damped_centerline_update(
            current_pole_cl, observed_pole_cl, step_size, smooth_tol,
            radii=radii, fixed_endpoints=True,
            update_beta=spine_update_beta)
        if pole_cl is None:
            break

        mesh = _build_integrated_pole_mesh(
            pole_cl, dt, contour_xy, pole_info, r_pole)
        if mesh is None:
            current_cl = pole_cl
            continue

        shift = _mean_centerline_shift(current_cl, _mesh_centers(mesh))
        valid = _validate_non_crossing_mesh(mesh)
        if valid and shift < best_shift:
            best_mesh = mesh
            best_r_pole = r_pole
            best_shift = shift
        if valid and shift < 0.25:
            return mesh, r_pole

        centers = _mesh_centers(mesh)
        next_cl = _damped_centerline_update(
            pole_cl, centers, step_size, smooth_tol, fixed_endpoints=True,
            update_beta=spine_update_beta)
        if next_cl is None:
            break
        current_cl = next_cl

    if best_mesh is not None:
        return best_mesh, best_r_pole
    return None


@njit(nogil=True, cache=True)
def _estimate_pole_radius_from_mesh(mesh):
    """Estimate pole radius as mean body half-width from contour ribs."""
    widths = np.sqrt((mesh[:, 0] - mesh[:, 2])**2 +
                     (mesh[:, 1] - mesh[:, 3])**2)
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if len(widths) == 0:
        return 0.0
    margin = max(2, int(round(0.1 * len(widths))))
    if len(widths) - 2 * margin >= 3:
        widths = widths[margin:-margin]
    return float(np.mean(widths / 2.0))


@njit(nogil=True, cache=True)
def _unit_vector(vec):
    vec = np.asarray(vec, dtype=np.float64)
    nrm = float(np.sqrt((vec * vec).sum()))
    if nrm < 1e-12 or not np.isfinite(nrm):
        return None
    return vec / nrm


@njit(nogil=True, cache=True)
def _point_in_polygon(point, polygon_xy):
    """Ray-casting point-in-polygon test for closed contours."""
    if polygon_xy is None or len(polygon_xy) < 4:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(len(polygon_xy) - 1):
        x0, y0 = polygon_xy[i]
        x1, y1 = polygon_xy[i + 1]
        if ((y0 > y) != (y1 > y)):
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0
            if x < x_cross:
                inside = not inside
    return inside


@njit(nogil=True, cache=True)
def _find_pole_tip(contour_xy, reference_point, outward_dir, r_pole):
    """Find the outward contour tip for one pole."""
    pts = contour_xy[:-1] if np.allclose(contour_xy[0], contour_xy[-1]) else contour_xy
    if pts is None or len(pts) < 3:
        return None
    rel = pts - reference_point
    axial = rel @ outward_dir
    if len(axial) == 0 or not np.any(np.isfinite(axial)):
        return None
    max_axial = float(np.max(axial))
    if max_axial <= 0:
        return None
    # Avoid selecting a distant contour segment on highly folded masks.
    if max_axial > max(3.0 * r_pole, r_pole + 12.0):
        return None
    normal = np.array([-outward_dir[1], outward_dir[0]])
    lateral = np.abs(rel @ normal)
    axial_window = max(0.5, 0.25 * r_pole)
    candidates = np.where(axial >= max_axial - axial_window)[0]
    if len(candidates) == 0:
        return pts[int(np.argmax(axial))]
    local_score = lateral[candidates] + 0.05 * (max_axial - axial[candidates])
    return pts[candidates[int(np.argmin(local_score))]]


@njit(nogil=True, cache=True)
def _set_endpoint_rib(mesh, end_idx, center, tangent, r_pole):
    tangent = _unit_vector(tangent)
    if tangent is None:
        return False
    normal = np.array([-tangent[1], tangent[0]])
    row = np.array([
        center[0] + normal[0] * r_pole,
        center[1] + normal[1] * r_pole,
        center[0] - normal[0] * r_pole,
        center[1] - normal[1] * r_pole,
    ])
    mesh[end_idx, :] = row
    return True


def _trim_endpoint_crossing_ribs(mesh):
    """Remove local body ribs that intersect anchored pole cap ribs."""
    if mesh is None or len(mesh) < 4:
        return mesh
    trimmed = mesh.copy()
    max_remove = max(2, int(0.15 * len(trimmed)))

    removed = 0
    while len(trimmed) > 3 and removed < max_remove:
        first = trimmed[0]
        crosses = any(
            _segments_intersect(first[0:2], first[2:4],
                                trimmed[j, 0:2], trimmed[j, 2:4])
            for j in range(1, len(trimmed))
        )
        if not crosses:
            break
        trimmed = np.vstack([trimmed[0:1], trimmed[2:]])
        removed += 1

    removed = 0
    while len(trimmed) > 3 and removed < max_remove:
        last = trimmed[-1]
        crosses = any(
            _segments_intersect(last[0:2], last[2:4],
                                trimmed[j, 0:2], trimmed[j, 2:4])
            for j in range(0, len(trimmed) - 1)
        )
        if not crosses:
            break
        trimmed = np.vstack([trimmed[:-2], trimmed[-1:]])
        removed += 1
    return trimmed


def _anchor_poles_to_contour_tips(mesh, contour_xy, r_pole):
    """Move endpoint rib centers to pole-tip-derived cap centers."""
    if (mesh is None or contour_xy is None or len(mesh) < 3 or
            not np.isfinite(r_pole) or r_pole <= 0):
        return None
    mesh_new = mesh.copy()
    centers = np.column_stack([
        (mesh[:, 0] + mesh[:, 2]) / 2.0,
        (mesh[:, 1] + mesh[:, 3]) / 2.0,
    ])

    first_dir = _unit_vector(centers[0] - centers[1])
    last_dir = _unit_vector(centers[-1] - centers[-2])
    if first_dir is None or last_dir is None:
        return None

    first_tip = _find_pole_tip(contour_xy, centers[0], first_dir, r_pole)
    last_tip = _find_pole_tip(contour_xy, centers[-1], last_dir, r_pole)
    if first_tip is None or last_tip is None:
        return None

    first_center = first_tip - first_dir * r_pole
    last_center = last_tip - last_dir * r_pole
    if (not _point_in_polygon(first_center, contour_xy) or
            not _point_in_polygon(last_center, contour_xy)):
        return None

    if not _set_endpoint_rib(
            mesh_new, 0, first_center, centers[1] - first_center, r_pole):
        return None
    if not _set_endpoint_rib(
            mesh_new, -1, last_center, last_center - centers[-2], r_pole):
        return None
    return _trim_endpoint_crossing_ribs(mesh_new)


@njit(nogil=True, cache=True)
def _signed_polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
    return 0.5 * area


@njit(nogil=True, cache=True)
def _point_on_segment(p, a, b, eps=1e-8):
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


@njit(nogil=True, cache=True)
def _segments_intersect(a, b, c, d, eps=1e-8):
    if (max(a[0], b[0]) + eps < min(c[0], d[0]) or
            max(c[0], d[0]) + eps < min(a[0], b[0]) or
            max(a[1], b[1]) + eps < min(c[1], d[1]) or
            max(c[1], d[1]) + eps < min(a[1], b[1])):
        return False
    o1 = _cross2(b - a, c - a)
    o2 = _cross2(b - a, d - a)
    o3 = _cross2(d - c, a - c)
    o4 = _cross2(d - c, b - c)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and _point_on_segment(c, a, b, eps):
        return True
    if abs(o2) <= eps and _point_on_segment(d, a, b, eps):
        return True
    if abs(o3) <= eps and _point_on_segment(a, c, d, eps):
        return True
    if abs(o4) <= eps and _point_on_segment(b, c, d, eps):
        return True
    return False


@njit(nogil=True, cache=True)
def _polyline_self_intersects(points):
    if points is None or len(points) < 4:
        return False
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        for j in range(i + 2, len(points) - 1):
            c, d = points[j], points[j + 1]
            if _segments_intersect(a, b, c, d):
                return True
    return False


@njit(nogil=True, cache=True)
def _mesh_has_rib_intersections(mesh):
    for i in range(len(mesh)):
        a = mesh[i, 0:2]
        b = mesh[i, 2:4]
        for j in range(i + 1, len(mesh)):
            c = mesh[j, 0:2]
            d = mesh[j, 2:4]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _validate_non_crossing_mesh(mesh, radii=None):
    """Validate that mesh ribs and side polylines are geometrically sane."""
    if mesh is None:
        return False
    radii_arr = np.empty(0, dtype=np.float64)
    has_radii = False
    if radii is not None:
        radii_arr = np.asarray(radii, dtype=np.float64)
        has_radii = True
    return bool(_validate_non_crossing_mesh_numba(
        np.asarray(mesh, dtype=np.float64), radii_arr, has_radii))


@njit(nogil=True, cache=True)
def _point_on_segment_xy(px, py, ax, ay, bx, by, eps=1e-8):
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps and
            min(ay, by) - eps <= py <= max(ay, by) + eps)


@njit(nogil=True, cache=True)
def _segments_intersect_xy(ax, ay, bx, by, cx, cy, dx, dy, eps=1e-8):
    if (max(ax, bx) + eps < min(cx, dx) or
            max(cx, dx) + eps < min(ax, bx) or
            max(ay, by) + eps < min(cy, dy) or
            max(cy, dy) + eps < min(ay, by)):
        return False

    o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and _point_on_segment_xy(cx, cy, ax, ay, bx, by, eps):
        return True
    if abs(o2) <= eps and _point_on_segment_xy(dx, dy, ax, ay, bx, by, eps):
        return True
    if abs(o3) <= eps and _point_on_segment_xy(ax, ay, cx, cy, dx, dy, eps):
        return True
    if abs(o4) <= eps and _point_on_segment_xy(bx, by, cx, cy, dx, dy, eps):
        return True
    return False


@njit(nogil=True, cache=True)
def _polyline_self_intersects_numba(points):
    n = len(points)
    if n < 4:
        return False
    for i in range(n - 1):
        ax = points[i, 0]
        ay = points[i, 1]
        bx = points[i + 1, 0]
        by = points[i + 1, 1]
        for j in range(i + 2, n - 1):
            if _segments_intersect_xy(
                    ax, ay, bx, by,
                    points[j, 0], points[j, 1],
                    points[j + 1, 0], points[j + 1, 1]):
                return True
    return False


@njit(nogil=True, cache=True)
def _mesh_has_rib_intersections_numba(mesh):
    n = len(mesh)
    for i in range(n):
        ax = mesh[i, 0]
        ay = mesh[i, 1]
        bx = mesh[i, 2]
        by = mesh[i, 3]
        for j in range(i + 1, n):
            if _segments_intersect_xy(
                    ax, ay, bx, by,
                    mesh[j, 0], mesh[j, 1], mesh[j, 2], mesh[j, 3]):
                return True
    return False


@njit(nogil=True, cache=True)
def _validate_non_crossing_mesh_numba(mesh, radii, has_radii):
    n = len(mesh)
    if n < 3:
        return False
    widths = np.zeros(n)
    for i in range(n):
        for j in range(4):
            if not np.isfinite(mesh[i, j]):
                return False
        dx = mesh[i, 0] - mesh[i, 2]
        dy = mesh[i, 1] - mesh[i, 3]
        width = np.sqrt(dx * dx + dy * dy)
        if not np.isfinite(width) or width < 0.5:
            return False
        widths[i] = width

    if has_radii and len(radii) == n:
        for i in range(n):
            expected = max(float(radii[i]), 0.5)
            half_width = widths[i] / 2.0
            if half_width < max(0.2, expected * 0.15):
                return False
            if half_width > max(expected * 2.5, expected + 4.0):
                return False

    if _mesh_has_rib_intersections_numba(mesh):
        return False
    if _polyline_self_intersects_numba(mesh[:, 0:2]):
        return False
    if _polyline_self_intersects_numba(mesh[:, 2:4]):
        return False

    first_sign = 0.0
    for i in range(n - 1):
        area = 0.5 * (
            mesh[i, 0] * mesh[i + 1, 1] -
            mesh[i + 1, 0] * mesh[i, 1] +
            mesh[i + 1, 0] * mesh[i + 1, 3] -
            mesh[i + 1, 2] * mesh[i + 1, 1] +
            mesh[i + 1, 2] * mesh[i, 3] -
            mesh[i, 2] * mesh[i + 1, 3] +
            mesh[i, 2] * mesh[i, 1] -
            mesh[i, 0] * mesh[i, 3])
        if abs(area) < 1e-3:
            return False
        sign = np.sign(area)
        if i == 0:
            first_sign = sign
        elif sign != first_sign:
            return False
    return True


@njit(nogil=True, cache=True)
def _optimize_pole_position(p_start, d_vec, r_pole, contour_pts, n_steps=30):
    """Find optimal pole position by fitting a semi-circle to the contour."""
    best_pos = p_start.copy()
    best_score = np.inf
    t_vals = np.linspace(-r_pole, r_pole, n_steps)
    angle = np.arctan2(d_vec[1], d_vec[0])
    arc_angles = np.linspace(angle - np.pi/2, angle + np.pi/2, 20)
    n_contour = len(contour_pts)

    for t_idx in range(n_steps):
        t = t_vals[t_idx]
        cx = p_start[0] + t * d_vec[0]
        cy = p_start[1] + t * d_vec[1]
        score = 0.0
        for k in range(20):
            ax = cx + r_pole * np.cos(arc_angles[k])
            ay = cy + r_pole * np.sin(arc_angles[k])
            min_d = np.inf
            for c in range(n_contour):
                dx = contour_pts[c, 0] - ax
                dy = contour_pts[c, 1] - ay
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_d:
                    min_d = d
            score += min_d
        score /= 20.0
        if score < best_score:
            best_score = score
            best_pos[0] = cx
            best_pos[1] = cy

    return best_pos


def _enforce_monotonic_mesh(mesh):
    """Re-sort ribs by PCA projection if centerline folds."""
    if len(mesh) < 3:
        return mesh
    cx = (mesh[:, 0] + mesh[:, 2]) / 2.0
    cy = (mesh[:, 1] + mesh[:, 3]) / 2.0
    cl = np.column_stack([cx, cy])
    d = np.diff(cl, axis=0)
    pl = float(np.sum(np.sqrt((d * d).sum(axis=1))))
    e2e = float(np.sqrt((cx[-1] - cx[0])**2 + (cy[-1] - cy[0])**2))
    if e2e > 1e-10 and pl / e2e > 2.5:
        centered = cl - cl.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        mesh = mesh[np.argsort(centered @ vh[0])]
    return mesh


# =============================================================================
# 4. Morphology Parameter Extraction
# =============================================================================

def extract_morphology(mesh, endpoint_radii=None):
    """Extract cell morphology parameters from the structured mesh.

    Cell length = spine_length + endpoint_radii[0] + endpoint_radii[-1].
    This accounts for the distance from skeleton endpoints to the poles.

    Parameters
    ----------
    mesh : np.ndarray (M, 4)
        [x_left, y_left, x_right, y_right] per rib.
    endpoint_radii : tuple (float, float) or None
        DT radii at first/last centerline points.

    Returns
    -------
    dict with keys: length, steplength, lengthvector, area, steparea,
        width, mean_width, max_width, volume, stepvolume, curvature,
        centerline (M, 2), n_ribs.
    """
    if mesh is None or len(mesh) < 2:
        return {}

    m = np.asarray(mesh, dtype=np.float64)
    has_endpoint_radii = endpoint_radii is not None
    r0 = 0.0
    r1 = 0.0
    if endpoint_radii is not None:
        r0 = float(endpoint_radii[0])
        r1 = float(endpoint_radii[1])
    (length, spine_length, steplength, lengthvector, area, steparea,
     width, mean_width, max_width, volume, stepvolume, curv,
     centerline) = _extract_morphology_numba(
        m, has_endpoint_radii, r0, r1)

    return dict(
        length=length,
        spine_length=spine_length,
        steplength=steplength,
        lengthvector=lengthvector,
        area=area,
        steparea=steparea,
        width=width,
        mean_width=mean_width,
        max_width=max_width,
        volume=volume,
        stepvolume=stepvolume,
        curvature=curv,
        centerline=centerline,
        n_ribs=len(m),
        pole_radius=r0 if has_endpoint_radii else None,
    )


@njit(nogil=True, cache=True)
def _extract_morphology_numba(m, has_endpoint_radii, r0, r1):
    """Numerical core for ``extract_morphology`` with unchanged formulas."""
    n_ribs = len(m)
    centerline = np.empty((n_ribs, 2), dtype=np.float64)
    for i in range(n_ribs):
        centerline[i, 0] = (m[i, 0] + m[i, 2]) / 2.0
        centerline[i, 1] = (m[i, 1] + m[i, 3]) / 2.0

    steplength = np.empty(n_ribs - 1, dtype=np.float64)
    for i in range(n_ribs - 1):
        dx = centerline[i + 1, 0] - centerline[i, 0]
        dy = centerline[i + 1, 1] - centerline[i, 1]
        steplength[i] = np.sqrt(dx * dx + dy * dy)
    spine_length = float(np.sum(steplength))

    lengthvector = np.empty(n_ribs, dtype=np.float64)
    lengthvector[0] = 0.0
    for i in range(1, n_ribs):
        lengthvector[i] = lengthvector[i - 1] + steplength[i - 1]
    if has_endpoint_radii:
        for i in range(n_ribs):
            lengthvector[i] += r0
        length = spine_length + r0 + r1
    else:
        length = spine_length

    width = np.empty(n_ribs, dtype=np.float64)
    for i in range(n_ribs):
        dx = m[i, 0] - m[i, 2]
        dy = m[i, 1] - m[i, 3]
        width[i] = np.sqrt(dx * dx + dy * dy)
    mean_width = float(np.mean(width))
    max_width = float(np.max(width))

    steparea = np.zeros(n_ribs - 1, dtype=np.float64)
    px = np.empty(4, dtype=np.float64)
    py = np.empty(4, dtype=np.float64)
    for i in range(n_ribs - 1):
        px[0] = m[i, 0]
        px[1] = m[i + 1, 0]
        px[2] = m[i + 1, 2]
        px[3] = m[i, 2]
        py[0] = m[i, 1]
        py[1] = m[i + 1, 1]
        py[2] = m[i + 1, 3]
        py[3] = m[i, 3]
        steparea[i] = _polygon_area(px, py)
    area = float(np.sum(steparea))
    if has_endpoint_radii:
        area += 0.5 * np.pi * (r0**2 + r1**2)

    stepvolume = np.zeros(n_ribs - 1, dtype=np.float64)
    for i in range(n_ribs - 1):
        d1 = width[i]
        d2 = width[i + 1]
        sl = steplength[i]
        stepvolume[i] = (
            d1 * d2 + (d1 - d2)**2 / 3.0) * sl * np.pi / 4.0
    volume = float(np.sum(stepvolume))
    if has_endpoint_radii:
        volume += (2.0 / 3.0) * np.pi * (r0**3 + r1**3)

    curvature = _compute_curvature(centerline)
    return (length, spine_length, steplength, lengthvector, area, steparea,
            width, mean_width, max_width, volume, stepvolume, curvature,
            centerline)


@njit(nogil=True, cache=True)
def _polygon_area(x, y):
    """Shoelace formula."""
    n = len(x)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += x[i] * y[j] - x[j] * y[i]
    return abs(a) / 2.0


@njit(nogil=True, cache=True)
def _gradient_1d(values):
    """Match NumPy's unit-spacing first-order 1D gradient."""
    n = len(values)
    gradient = np.empty(n, dtype=np.float64)
    if n == 1:
        gradient[0] = 0.0
        return gradient
    gradient[0] = values[1] - values[0]
    for i in range(1, n - 1):
        gradient[i] = (values[i + 1] - values[i - 1]) / 2.0
    gradient[n - 1] = values[n - 1] - values[n - 2]
    return gradient


@njit(nogil=True, cache=True)
def _compute_curvature(centerline):
    """Mean absolute curvature of the centerline."""
    if len(centerline) < 3:
        return 0.0
    dx = _gradient_1d(centerline[:, 0])
    dy = _gradient_1d(centerline[:, 1])
    ddx = _gradient_1d(dx)
    ddy = _gradient_1d(dy)
    total = 0.0
    finite_count = 0
    for i in range(len(centerline)):
        numerator = abs(dx[i] * ddy[i] - dy[i] * ddx[i])
        denominator = (dx[i] * dx[i] + dy[i] * dy[i] + 1e-10) ** 1.5
        curvature = numerator / denominator
        if curvature > 10.0:
            curvature = 10.0
        if np.isfinite(curvature):
            total += curvature
            finite_count += 1
    if finite_count == 0:
        return np.nan
    return float(total / finite_count)


# =============================================================================
# 5. Pipeline
# =============================================================================

def process_mask(mask, step_size=None, smooth_tol=0.001):
    """Process all labeled cells in a mask.

    Parameters
    ----------
    mask : np.ndarray, 2D integer
        Labeled mask (0=background, >0 = cells).
    step_size : float or None
    smooth_tol : float

    Returns
    -------
    list of dict : one per cell.
    """
    labels = np.unique(mask[mask > 0])
    results = []
    for label in sorted(labels):
        cell_mask = (mask == label)
        mesh, ep_radii = dt_medial_axis(cell_mask, step_size=step_size,
                                         smooth_tol=smooth_tol)
        if mesh is None:
            results.append({
                'label': int(label),
                'mesh': None,
                'error': 'Medial axis failed',
            })
        else:
            morph = extract_morphology(mesh, endpoint_radii=ep_radii)
            morph['label'] = int(label)
            morph['mesh'] = mesh
            results.append(morph)
    return results


def process_mask_file(path, **kwargs):
    """Load a mask file and process all cells."""
    from skimage.io import imread
    mask = imread(path)
    return process_mask(mask, **kwargs)


def results_to_dataframe(results):
    """Convert morphology results to a pandas DataFrame."""
    import pandas as pd
    rows = []
    for r in results:
        row = {
            'label': r.get('label', -1),
            'length': r.get('length', np.nan),
            'spine_length': r.get('spine_length', np.nan),
            'mean_width': r.get('mean_width', np.nan),
            'max_width': r.get('max_width', np.nan),
            'area': r.get('area', np.nan),
            'volume': r.get('volume', np.nan),
            'curvature': r.get('curvature', np.nan),
            'n_ribs': r.get('n_ribs', 0),
            'error': r.get('error', None),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6. Visualization
# =============================================================================

def plot_cell_mesh(ax, morph, color='lime', linewidth=0.8):
    """Plot cell mesh ribs and centerline on a matplotlib axis."""
    mesh = morph.get('mesh')
    if mesh is None or len(mesh) < 2:
        return
    for i in range(len(mesh)):
        ax.plot([mesh[i, 0], mesh[i, 2]], [mesh[i, 1], mesh[i, 3]],
                color=color, linewidth=linewidth)
    cl = morph.get('centerline')
    if cl is not None:
        ax.plot(cl[:, 0], cl[:, 1], color='cyan', linewidth=linewidth * 1.5)


# =============================================================================
# 7. Quick test
# =============================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r'cells_mask\20251014_233501_fov_1_channel_2_t_50_masks.tif'

    results = process_mask_file(path)
    df = results_to_dataframe(results)
    print(df.to_string())
    n_ok = sum(1 for r in results if r.get('error') is None)
    print('\nProcessed {} cells, {} succeeded.'.format(len(results), n_ok))
