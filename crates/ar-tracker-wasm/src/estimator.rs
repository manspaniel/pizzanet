//! Metric state estimation on top of the LK front-end, built on the `arael`
//! nonlinear least-squares framework (compile-time symbolic differentiation,
//! sparse Levenberg-Marquardt).
//!
//! Two solvers:
//! - [`solve_frame_pose`]: per-frame 3-DOF camera position refinement against
//!   the current landmark map, rotation held fixed at the IMU value, with a
//!   soft prior toward the inertial prediction and one trimming re-solve for
//!   outlier rejection.
//! - [`solve_window`]: sliding-window visual-inertial bundle adjustment —
//!   keyframe positions and velocities plus per-landmark inverse depths,
//!   coupled by pixel reprojection factors and world-frame accelerometer
//!   preintegration factors. The preintegration factors are what make the
//!   monocular reconstruction metric: vision fixes the shape, the integrated
//!   accelerometer fixes the scale.
//!
//! Rotations are never optimized: iOS sensor fusion supplies orientation with
//! better short-term fidelity than anything recoverable from these frames, so
//! every factor bakes its rotation in as constants. This removes SO(3)
//! parameters, the rotation gauge, and most of the linearization cost.

use crate::geometry::Intrinsics;
use crate::map::{MAX_INVERSE_DEPTH, MIN_INVERSE_DEPTH, Map, Preintegration};
use arael::model::{CrossBlock, Param, SelfBlock, TripletBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::vect3f;
use glam::DMat3;
use vio_core::{DQuat, DVec3};

/// Reprojection residual weight: 1 / (pixel sigma).
const REPROJECTION_ISIGMA: f32 = 1.0 / 1.5;
/// Clamp range for the adaptive per-frame outlier threshold (pixels). The
/// threshold tracks the median residual so unconverged landmark depths (which
/// misfit systematically until bundle adjustment learns them) are not mass
/// rejected.
const FRAME_OUTLIER_MIN_PIXELS: f64 = 2.0;
const FRAME_OUTLIER_MAX_PIXELS: f64 = 8.0;
const WINDOW_OUTLIER_PIXELS: f64 = 5.0;
/// Preintegration weights. Phone accelerometers are noisy and the intervals
/// short, so these act as a scale-observing hint, not a hard constraint.
const PREINT_VELOCITY_ISIGMA: f32 = 1.0 / 0.35;
const PREINT_POSITION_ISIGMA: f32 = 1.0 / 0.18;
/// Weak inverse-depth prior keeping unobserved depths near initialization
/// until parallax takes over. Once a landmark's parallax has converged its
/// depth, the prior all but releases — otherwise the depth priors collectively
/// decide the scale gauge (reprojection factors are scale-free), and the map
/// stays glued to the initialization depth instead of the accelerometer's
/// metric scale.
const INVERSE_DEPTH_PRIOR_ISIGMA: f32 = 1.0 / 0.35;
/// Gauge prior pinning the oldest window keyframe (approximate
/// marginalization: information from factors that slid out of the window is
/// summarized as trust in that keyframe's current estimate).
const GAUGE_POSITION_ISIGMA: f32 = 1.0 / 0.01;
/// Weak prior tying every other window keyframe to its current estimate. This
/// keeps the solve non-degenerate when a keyframe temporarily has no usable
/// reprojection factors, and bounds how far one solve may move the past.
const WINDOW_POSITION_ISIGMA: f32 = 1.0 / 0.4;
const OLDEST_VELOCITY_ISIGMA: f32 = 1.0 / 0.2;
/// Per-frame prior pulling the camera toward the inertial prediction.
const FRAME_PRIOR_ISIGMA: f32 = 1.0 / 0.25;

/// Cap on landmarks entering one window solve, best-observed first.
const MAX_WINDOW_LANDMARKS: usize = 180;

fn vec3_to_f32(v: DVec3) -> vect3f {
    vect3f::new(v.x as f32, v.y as f32, v.z as f32)
}

fn vec3_from_f32(v: vect3f) -> DVec3 {
    DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
}

/// Rows of the world-to-camera rotation matrix (`R^T`) as three constants.
fn world_to_camera_rows(orientation: DQuat) -> [vect3f; 3] {
    let matrix = DMat3::from_quat(orientation.conjugate());
    // glam DMat3 is column-major; row i = (col0[i], col1[i], col2[i]).
    let row = |i: usize| {
        vect3f::new(
            matrix.col(0)[i] as f32,
            matrix.col(1)[i] as f32,
            matrix.col(2)[i] as f32,
        )
    };
    [row(0), row(1), row(2)]
}

// ---------------------------------------------------------------------------
// Per-frame pose model
// ---------------------------------------------------------------------------

#[arael::model]
#[arael(constraint(hb, {
    let d = camnode.pos - camnode.prior_pos;
    [d.x * camnode.prior_isigma, d.y * camnode.prior_isigma, d.z * camnode.prior_isigma]
}))]
struct CamNode {
    pos: Param<vect3f>,
    prior_pos: vect3f,
    prior_isigma: f32,
    hb: SelfBlock<CamNode, f32>,
}

// Reprojection of one fixed-world-position landmark into the single camera,
// rotation baked in. Follows the loc_demo containment shape: the observation
// constraint lives inside the detail entity that carries its constants, and
// writes into the camera's Hessian block through the ref (remote block).
#[arael::model]
#[arael(constraint(cam.hb, parent = detail, {
    let d = detail.lm_world - cam.pos;
    let cx = detail.rb0.x * d.x + detail.rb0.y * d.y + detail.rb0.z * d.z;
    let cy = detail.rb1.x * d.x + detail.rb1.y * d.y + detail.rb1.z * d.z;
    let cz = detail.rb2.x * d.x + detail.rb2.y * d.y + detail.rb2.z * d.z;
    let depth = 0.0 - cz;
    [(detail.focal * cx / depth - detail.pixel_x) * detail.isigma,
     (detail.focal * (0.0 - cy) / depth - detail.pixel_y) * detail.isigma]
}))]
struct CamObs {
    #[arael(ref = root.cameras)]
    cam: Ref<CamNode>,
}

// Constants of one observation: the landmark's fixed world position, the
// frame's fixed world-to-camera rotation rows, intrinsics, and the measured
// pixel. A plain entity (no params) holding its constraint struct, as in
// arael's loc_demo — remote-block constraints cannot read their own fields.
#[arael::model]
struct ObsDetail {
    lm_world: vect3f,
    rb0: vect3f,
    rb1: vect3f,
    rb2: vect3f,
    focal: f32,
    pixel_x: f32,
    pixel_y: f32,
    isigma: f32,
    frines: std::vec::Vec<CamObs>,
}

#[arael::model]
#[arael(root, f32)]
struct FrameProblem {
    cameras: refs::Vec<CamNode>,
    details: refs::Vec<ObsDetail>,
}

/// One landmark's contribution to the per-frame solve.
pub struct FrameObservation {
    pub landmark: u32,
    pub world: DVec3,
    /// Centered pixel coordinates (pixel minus principal point).
    pub pixel_x: f64,
    pub pixel_y: f64,
}

pub struct FramePoseSolution {
    pub position: DVec3,
    pub matches: usize,
    pub inliers: usize,
    pub outlier_landmarks: Vec<u32>,
}

/// Refines the camera position from landmark reprojections. `orientation` is
/// the IMU camera orientation for this frame; `predicted` the inertial
/// position prediction used both as the initial value and as a soft prior.
pub fn solve_frame_pose(
    observations: &[FrameObservation],
    orientation: DQuat,
    predicted: DVec3,
    intrinsics: &Intrinsics,
) -> Option<FramePoseSolution> {
    if observations.len() < 4 {
        return None;
    }
    let rows = world_to_camera_rows(orientation);
    let build = |subset: &[&FrameObservation], prior_isigma: f32| -> FrameProblem {
        let mut problem = FrameProblem {
            cameras: refs::Vec::new(),
            details: refs::Vec::new(),
        };
        problem.cameras.push(CamNode {
            pos: Param::new(vec3_to_f32(predicted)),
            prior_pos: vec3_to_f32(predicted),
            prior_isigma,
            hb: SelfBlock::new(),
        });
        let cam = problem.cameras.ref_at(0);
        for observation in subset {
            problem.details.push(ObsDetail {
                lm_world: vec3_to_f32(observation.world),
                rb0: rows[0],
                rb1: rows[1],
                rb2: rows[2],
                focal: intrinsics.focal as f32,
                pixel_x: observation.pixel_x as f32,
                pixel_y: observation.pixel_y as f32,
                isigma: REPROJECTION_ISIGMA,
                frines: std::vec::Vec::from([CamObs { cam }]),
            });
        }
        problem
    };
    let solve = |problem: &mut FrameProblem| -> DVec3 {
        let config = LmConfig::<f32> {
            max_iters: 10,
            ..Default::default()
        };
        let _ = problem.solve_sparse(&config);
        vec3_from_f32(problem.cameras[0].pos.value)
    };
    let residual_pixels = |observation: &FrameObservation, position: DVec3| -> f64 {
        let camera = orientation.conjugate() * (observation.world - position);
        let depth = -camera.z;
        if depth <= 0.05 {
            return f64::INFINITY;
        }
        let u = intrinsics.focal * camera.x / depth;
        let v = intrinsics.focal * -camera.y / depth;
        (u - observation.pixel_x).hypot(v - observation.pixel_y)
    };

    let all: Vec<&FrameObservation> = observations.iter().collect();
    let mut problem = build(&all, FRAME_PRIOR_ISIGMA);
    let first_pass = solve(&mut problem);

    let residuals: Vec<f64> = observations
        .iter()
        .map(|observation| residual_pixels(observation, first_pass))
        .collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let threshold = (median * 2.5).clamp(FRAME_OUTLIER_MIN_PIXELS, FRAME_OUTLIER_MAX_PIXELS);
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("AR_DEBUG_FRAME").is_some() {
        let p10 = sorted.first().copied().unwrap_or(0.0);
        let p90 = sorted
            .get(sorted.len() * 9 / 10)
            .copied()
            .unwrap_or(f64::NAN);
        let mut at_prediction: Vec<f64> = observations
            .iter()
            .map(|observation| residual_pixels(observation, predicted))
            .collect();
        at_prediction.sort_by(f64::total_cmp);
        let start = at_prediction
            .get(at_prediction.len() / 2)
            .copied()
            .unwrap_or(0.0);
        eprintln!(
            "frame-solve obs={} med={:.2} p90={:.2} min={:.2} thr={:.2} pred_med={:.2} step={:.3}",
            observations.len(),
            median,
            p90,
            p10,
            threshold,
            start,
            (first_pass - predicted).length(),
        );
    }
    let inlier_refs: Vec<&FrameObservation> = observations
        .iter()
        .zip(&residuals)
        .filter(|(_, residual)| **residual <= threshold)
        .map(|(observation, _)| observation)
        .collect();
    let outlier_landmarks: Vec<u32> = observations
        .iter()
        .zip(&residuals)
        .filter(|(_, residual)| **residual > threshold)
        .map(|(observation, _)| observation.landmark)
        .collect();
    if inlier_refs.len() < 4 {
        return Some(FramePoseSolution {
            position: predicted,
            matches: observations.len(),
            inliers: inlier_refs.len(),
            outlier_landmarks,
        });
    }
    let position = if outlier_landmarks.is_empty() {
        first_pass
    } else {
        let mut trimmed = build(&inlier_refs, FRAME_PRIOR_ISIGMA);
        solve(&mut trimmed)
    };
    if !position.is_finite() {
        return None;
    }
    Some(FramePoseSolution {
        position,
        matches: observations.len(),
        inliers: inlier_refs.len(),
        outlier_landmarks,
    })
}

// ---------------------------------------------------------------------------
// Sliding-window visual-inertial bundle adjustment
// ---------------------------------------------------------------------------

#[arael::model]
#[arael(constraint(hb, {
    let dp = kfnode.pos - kfnode.prior_pos;
    let dv = kfnode.vel - kfnode.prior_vel;
    [dp.x * kfnode.prior_pos_isigma, dp.y * kfnode.prior_pos_isigma, dp.z * kfnode.prior_pos_isigma,
     dv.x * kfnode.prior_vel_isigma, dv.y * kfnode.prior_vel_isigma, dv.z * kfnode.prior_vel_isigma]
}))]
struct KfNode {
    pos: Param<vect3f>,
    vel: Param<vect3f>,
    prior_pos: vect3f,
    prior_vel: vect3f,
    prior_pos_isigma: f32,
    prior_vel_isigma: f32,
    hb: SelfBlock<KfNode, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(lmnode.rho - lmnode.prior_rho) * lmnode.prior_isigma]
}))]
struct LmNode {
    rho: Param<f32>,
    prior_rho: f32,
    prior_isigma: f32,
    hb: SelfBlock<LmNode, f32>,
}

// Reprojection with the landmark's anchor keyframe inside the window: the
// world point is anchor.pos + dir_anchor / rho, projected into the observer
// with its fixed rotation. Couples three entities (anchor pos, observer pos,
// inverse depth) — a TripletBlock factor; arael's Schur path knows this
// pose–inverse-depth–pose shape.
#[arael::model]
#[arael(constraint(hb, {
    let inv = 1.0 / lm.rho;
    let px = anchor.pos.x + reprojwindow.dir_anchor.x * inv - observer.pos.x;
    let py = anchor.pos.y + reprojwindow.dir_anchor.y * inv - observer.pos.y;
    let pz = anchor.pos.z + reprojwindow.dir_anchor.z * inv - observer.pos.z;
    let cx = reprojwindow.rb0.x * px + reprojwindow.rb0.y * py + reprojwindow.rb0.z * pz;
    let cy = reprojwindow.rb1.x * px + reprojwindow.rb1.y * py + reprojwindow.rb1.z * pz;
    let cz = reprojwindow.rb2.x * px + reprojwindow.rb2.y * py + reprojwindow.rb2.z * pz;
    let depth = 0.0 - cz;
    [(reprojwindow.focal * cx / depth - reprojwindow.pixel_x) * reprojwindow.isigma,
     (reprojwindow.focal * (0.0 - cy) / depth - reprojwindow.pixel_y) * reprojwindow.isigma]
}))]
struct ReprojWindow {
    #[arael(ref = root.keyframes)]
    anchor: Ref<KfNode>,
    #[arael(ref = root.keyframes)]
    observer: Ref<KfNode>,
    #[arael(ref = root.landmarks)]
    lm: Ref<LmNode>,
    /// Anchor rotation times the landmark bearing (constant).
    dir_anchor: vect3f,
    rb0: vect3f,
    rb1: vect3f,
    rb2: vect3f,
    focal: f32,
    pixel_x: f32,
    pixel_y: f32,
    isigma: f32,
    hb: TripletBlock<f32>,
}

// Reprojection whose anchor keyframe has left the window: the anchor position
// is a constant, so only (observer pos, inverse depth) couple.
#[arael::model]
#[arael(constraint(hb, {
    let inv = 1.0 / lm.rho;
    let px = reprojfrozen.anchor_pos.x + reprojfrozen.dir_anchor.x * inv - observer.pos.x;
    let py = reprojfrozen.anchor_pos.y + reprojfrozen.dir_anchor.y * inv - observer.pos.y;
    let pz = reprojfrozen.anchor_pos.z + reprojfrozen.dir_anchor.z * inv - observer.pos.z;
    let cx = reprojfrozen.rb0.x * px + reprojfrozen.rb0.y * py + reprojfrozen.rb0.z * pz;
    let cy = reprojfrozen.rb1.x * px + reprojfrozen.rb1.y * py + reprojfrozen.rb1.z * pz;
    let cz = reprojfrozen.rb2.x * px + reprojfrozen.rb2.y * py + reprojfrozen.rb2.z * pz;
    let depth = 0.0 - cz;
    [(reprojfrozen.focal * cx / depth - reprojfrozen.pixel_x) * reprojfrozen.isigma,
     (reprojfrozen.focal * (0.0 - cy) / depth - reprojfrozen.pixel_y) * reprojfrozen.isigma]
}))]
struct ReprojFrozen {
    #[arael(ref = root.keyframes)]
    observer: Ref<KfNode>,
    #[arael(ref = root.landmarks)]
    lm: Ref<LmNode>,
    anchor_pos: vect3f,
    dir_anchor: vect3f,
    rb0: vect3f,
    rb1: vect3f,
    rb2: vect3f,
    focal: f32,
    pixel_x: f32,
    pixel_y: f32,
    isigma: f32,
    hb: CrossBlock<KfNode, LmNode, f32>,
}

// World-frame accelerometer preintegration between consecutive keyframes:
// v_j = v_i + ∫a, p_j = p_i + v_i·dt + ∫∫a. Gravity-removed, bias-corrected,
// rotated per-sample by the IMU orientation before integration. This is the
// metric-scale anchor for the monocular reconstruction.
#[arael::model]
#[arael(constraint(hb, {
    let rv_x = cur.vel.x - prev.vel.x - preintpair.dv.x;
    let rv_y = cur.vel.y - prev.vel.y - preintpair.dv.y;
    let rv_z = cur.vel.z - prev.vel.z - preintpair.dv.z;
    let rp_x = cur.pos.x - prev.pos.x - prev.vel.x * preintpair.dt - preintpair.dp.x;
    let rp_y = cur.pos.y - prev.pos.y - prev.vel.y * preintpair.dt - preintpair.dp.y;
    let rp_z = cur.pos.z - prev.pos.z - prev.vel.z * preintpair.dt - preintpair.dp.z;
    [rv_x * preintpair.isigma_v, rv_y * preintpair.isigma_v, rv_z * preintpair.isigma_v,
     rp_x * preintpair.isigma_p, rp_y * preintpair.isigma_p, rp_z * preintpair.isigma_p]
}))]
struct PreintPair {
    #[arael(ref = root.keyframes)]
    prev: Ref<KfNode>,
    #[arael(ref = root.keyframes)]
    cur: Ref<KfNode>,
    dt: f32,
    dv: vect3f,
    dp: vect3f,
    isigma_v: f32,
    isigma_p: f32,
    hb: CrossBlock<KfNode, KfNode, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WindowProblem {
    keyframes: refs::Vec<KfNode>,
    landmarks: refs::Vec<LmNode>,
    window_reprojections: refs::Vec<ReprojWindow>,
    frozen_reprojections: refs::Vec<ReprojFrozen>,
    preintegrations: refs::Vec<PreintPair>,
}

#[allow(dead_code)] // diagnostic fields surface via replay tooling
pub struct WindowSolveReport {
    pub keyframes: usize,
    pub landmarks: usize,
    pub reprojections: usize,
    pub iterations: usize,
    pub start_cost: f64,
    pub end_cost: f64,
}

/// Runs sliding-window bundle adjustment over the map's current window and
/// writes refined keyframe positions/velocities and landmark inverse depths
/// back into the map. Returns `None` when the window has fewer than two
/// keyframes or no usable observations.
pub fn solve_window(map: &mut Map, intrinsics: &Intrinsics) -> Option<WindowSolveReport> {
    let window = map.window_ids();
    if window.len() < 2 {
        return None;
    }

    // Landmarks observed by at least two window keyframes (or anchored outside
    // with one window observation), capped, best-observed first.
    let mut candidate_landmarks: Vec<(u32, u32, f64)> = map
        .landmarks
        .iter()
        .filter(|landmark| landmark.observation_count >= 2)
        .map(|landmark| {
            (
                landmark.id,
                landmark.observation_count,
                landmark.max_parallax_degrees,
            )
        })
        .collect();
    candidate_landmarks.sort_by(|a, b| {
        (b.1, b.2.total_cmp(&a.2) as i8).cmp(&(a.1, 0)).then(
            b.2.total_cmp(&a.2),
        )
    });
    candidate_landmarks.truncate(MAX_WINDOW_LANDMARKS);
    let selected: Vec<u32> = candidate_landmarks.iter().map(|(id, _, _)| *id).collect();
    if selected.is_empty() {
        return None;
    }

    // Native-only ablation switches for offline replay experiments.
    #[cfg(not(target_arch = "wasm32"))]
    let (disable_ba, disable_preint) = (
        std::env::var_os("AR_DISABLE_BA").is_some(),
        std::env::var_os("AR_DISABLE_PREINT").is_some(),
    );
    #[cfg(target_arch = "wasm32")]
    let (disable_ba, disable_preint) = (false, false);
    if disable_ba {
        return None;
    }

    let mut problem = WindowProblem {
        keyframes: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        window_reprojections: refs::Vec::new(),
        frozen_reprojections: refs::Vec::new(),
        preintegrations: refs::Vec::new(),
    };

    // Keyframe nodes. The oldest window keyframe carries a tight position
    // prior (gauge + approximate marginalization); every node carries a weak
    // velocity prior toward its current estimate to keep the inertial states
    // bounded when accelerometer information is thin.
    let mut node_index: Vec<(u32, usize)> = Vec::with_capacity(window.len());
    for (index, keyframe_id) in window.iter().enumerate() {
        let keyframe = map.keyframe(*keyframe_id)?;
        problem.keyframes.push(KfNode {
            pos: Param::new(vec3_to_f32(keyframe.position)),
            vel: Param::new(vec3_to_f32(keyframe.velocity)),
            prior_pos: vec3_to_f32(keyframe.position),
            prior_vel: vec3_to_f32(keyframe.velocity),
            prior_pos_isigma: if index == 0 {
                GAUGE_POSITION_ISIGMA
            } else {
                WINDOW_POSITION_ISIGMA
            },
            prior_vel_isigma: if index == 0 {
                OLDEST_VELOCITY_ISIGMA
            } else {
                OLDEST_VELOCITY_ISIGMA * 0.25
            },
            hb: SelfBlock::new(),
        });
        node_index.push((*keyframe_id, index));
    }
    let node_of = |keyframe_id: u32| -> Option<usize> {
        node_index
            .iter()
            .find(|(id, _)| *id == keyframe_id)
            .map(|(_, index)| *index)
    };

    // Landmark nodes.
    let mut landmark_index: Vec<(u32, usize)> = Vec::with_capacity(selected.len());
    for (index, landmark_id) in selected.iter().enumerate() {
        let landmark = map.landmark(*landmark_id)?;
        problem.landmarks.push(LmNode {
            rho: Param::new(landmark.inverse_depth as f32),
            prior_rho: landmark.inverse_depth as f32,
            prior_isigma: INVERSE_DEPTH_PRIOR_ISIGMA,
            hb: SelfBlock::new(),
        });
        landmark_index.push((*landmark_id, index));
    }
    let landmark_node_of = |landmark_id: u32| -> Option<usize> {
        landmark_index
            .iter()
            .find(|(id, _)| *id == landmark_id)
            .map(|(_, index)| *index)
    };

    // Reprojection factors from every window keyframe's stored observations.
    let mut reprojection_count = 0usize;
    for keyframe_id in &window {
        let keyframe = map.keyframe(*keyframe_id)?.clone();
        let observer_node = node_of(*keyframe_id)?;
        let rows = world_to_camera_rows(keyframe.orientation);
        for observation in &keyframe.observations {
            let Some(landmark_node) = landmark_node_of(observation.landmark) else {
                continue;
            };
            let landmark = map.landmark(observation.landmark)?;
            if landmark.anchor == *keyframe_id {
                // Observing from the anchor adds no depth information — the
                // bearing was defined there.
                continue;
            }
            let Some(anchor) = map.keyframe(landmark.anchor) else {
                continue;
            };
            let dir_anchor = vec3_to_f32(anchor.orientation * landmark.bearing);
            let pixel_x = (f64::from(observation.pixel.0) - intrinsics.center_x) as f32;
            let pixel_y = (f64::from(observation.pixel.1) - intrinsics.center_y) as f32;
            // Verify the current estimate reprojects sanely; wildly wrong
            // observations (moving objects, association errors) are excluded
            // from this solve rather than robustified.
            if let Some(world) = map.landmark_world(landmark) {
                let camera = keyframe.orientation.conjugate() * (world - keyframe.position);
                let depth = -camera.z;
                if depth <= 0.05 {
                    continue;
                }
                let u = intrinsics.focal * camera.x / depth;
                let v = intrinsics.focal * -camera.y / depth;
                if (u - f64::from(pixel_x)).hypot(v - f64::from(pixel_y))
                    > WINDOW_OUTLIER_PIXELS * 8.0
                {
                    continue;
                }
            }
            if let Some(anchor_node) = node_of(landmark.anchor) {
                problem.window_reprojections.push(ReprojWindow {
                    anchor: problem.keyframes.ref_at(anchor_node),
                    observer: problem.keyframes.ref_at(observer_node),
                    lm: problem.landmarks.ref_at(landmark_node),
                    dir_anchor,
                    rb0: rows[0],
                    rb1: rows[1],
                    rb2: rows[2],
                    focal: intrinsics.focal as f32,
                    pixel_x,
                    pixel_y,
                    isigma: REPROJECTION_ISIGMA,
                    hb: TripletBlock::new(),
                });
            } else {
                problem.frozen_reprojections.push(ReprojFrozen {
                    observer: problem.keyframes.ref_at(observer_node),
                    lm: problem.landmarks.ref_at(landmark_node),
                    anchor_pos: vec3_to_f32(anchor.position),
                    dir_anchor,
                    rb0: rows[0],
                    rb1: rows[1],
                    rb2: rows[2],
                    focal: intrinsics.focal as f32,
                    pixel_x,
                    pixel_y,
                    isigma: REPROJECTION_ISIGMA,
                    hb: CrossBlock::new(),
                });
            }
            reprojection_count += 1;
        }
    }
    if reprojection_count < 8 {
        return None;
    }

    // Preintegration factors between consecutive window keyframes.
    for pair in window.windows(2) {
        if disable_preint {
            break;
        }
        let current = map.keyframe(pair[1])?;
        let Some(preintegration) = current.preintegration else {
            continue;
        };
        if preintegration.sample_count < 4 || preintegration.duration_seconds <= 0.01 {
            continue;
        }
        let (Some(prev_node), Some(cur_node)) = (node_of(pair[0]), node_of(pair[1])) else {
            continue;
        };
        problem.preintegrations.push(PreintPair {
            prev: problem.keyframes.ref_at(prev_node),
            cur: problem.keyframes.ref_at(cur_node),
            dt: preintegration.duration_seconds as f32,
            dv: vec3_to_f32(preintegration.delta_velocity),
            dp: vec3_to_f32(preintegration.delta_position),
            isigma_v: PREINT_VELOCITY_ISIGMA,
            isigma_p: PREINT_POSITION_ISIGMA,
            hb: CrossBlock::new(),
        });
    }

    let config = LmConfig::<f32> {
        max_iters: 12,
        ..Default::default()
    };
    // Dense solve: the window is small (~220 params) and the 0.7 sparse
    // indexed assembly does not cover symbolic TripletBlock factors.
    let mut initial = std::vec::Vec::new();
    problem.serialize32(&mut initial);
    let result = arael::simple_lm::solve_f32(&initial, &mut problem, &config);
    if !result.end_cost.is_finite() {
        return None;
    }
    problem.deserialize32(&result.x);

    // Write back.
    for (keyframe_id, node) in &node_index {
        let solved = &problem.keyframes[*node];
        let position = vec3_from_f32(solved.pos.value);
        let velocity = vec3_from_f32(solved.vel.value);
        if let Some(keyframe) = map
            .keyframes
            .iter_mut()
            .find(|keyframe| keyframe.id == *keyframe_id)
            && position.is_finite()
            && velocity.is_finite()
        {
            keyframe.position = position;
            keyframe.velocity = velocity;
        }
    }
    for (landmark_id, node) in &landmark_index {
        let solved = f64::from(problem.landmarks[*node].rho.value);
        if let Some(landmark) = map.landmark_mut(*landmark_id)
            && solved.is_finite()
        {
            landmark.inverse_depth = solved.clamp(MIN_INVERSE_DEPTH, MAX_INVERSE_DEPTH);
        }
    }

    Some(WindowSolveReport {
        keyframes: window.len(),
        landmarks: selected.len(),
        reprojections: reprojection_count,
        iterations: result.iterations,
        start_cost: f64::from(result.start_cost),
        end_cost: f64::from(result.end_cost),
    })
}

/// Convenience: preintegration accessor used by lib.rs when sealing a
/// keyframe.
pub fn preintegration_is_usable(preintegration: &Preintegration) -> bool {
    preintegration.sample_count >= 4
        && preintegration.duration_seconds > 0.01
        && preintegration.delta_velocity.is_finite()
        && preintegration.delta_position.is_finite()
}
