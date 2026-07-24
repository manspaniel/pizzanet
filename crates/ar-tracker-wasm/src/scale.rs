//! Closed-form metric-scale estimation from preintegrated acceleration.
//!
//! A monocular map is correct only up to scale: shrinking every translation
//! and depth together fits the images equally well, so something metric must
//! pick the gauge. The accelerometer is that something — it measures m/s²
//! identically on every device. Over a chain of keyframes with visual
//! positions `p_i` (map units) and world-frame preintegrated increments
//! `Δv_i, Δp_i` (metric), the kinematics
//!
//! `s·(p_{i+1} − p_i) = v_i·dt_i + Δp_i`,  `v_{i+1} = v_i + Δv_i`
//!
//! are linear in the unknowns `(s, v_0)`; four unknowns, three equations per
//! pair, solved in closed form by 4×4 normal equations. The estimate is only
//! trusted when the chain contains genuine acceleration excitation (direction
//! changes) — during perfectly smooth motion the velocity states absorb any
//! scale and the system is unobservable, which is exactly when we must not
//! touch the gauge.

use crate::map::Map;
use vio_core::DVec3;

/// Minimum contiguous keyframe pairs before an estimate is attempted.
const MIN_PAIRS: usize = 5;
/// Minimum summed |Δv| across the chain (m/s) for the scale to be observable.
const MIN_EXCITATION_MPS: f64 = 0.5;
/// Minimum chain duration in seconds.
const MIN_SPAN_SECONDS: f64 = 1.5;
/// Estimates outside this ratio band are treated as failed solves.
const MIN_RATIO: f64 = 0.1;
const MAX_RATIO: f64 = 6.0;

/// One scale observation: the multiplicative correction that would make the
/// map metric, and how observable it was.
#[derive(Clone, Copy, Debug)]
pub struct ScaleEstimate {
    /// Multiply map translations/depth by this to make them metric.
    pub ratio: f64,
    /// Summed |Δv| over the chain, m/s — the excitation that made the ratio
    /// observable.
    pub excitation: f64,
    /// Keyframe pairs in the solved chain (diagnostic).
    #[allow(dead_code)]
    pub pairs: usize,
}

/// Estimates the metric-scale correction from the map's most recent contiguous
/// chain of preintegrated keyframe pairs. Returns `None` when the chain is too
/// short, too calm, or the solve is degenerate.
pub fn estimate_scale(map: &Map) -> Option<ScaleEstimate> {
    // Collect the longest contiguous run of consecutive keyframes carrying
    // preintegration, ending at the newest keyframe.
    let mut pairs: Vec<(DVec3, f64, DVec3, DVec3)> = Vec::new();
    for window in map.keyframes.windows(2).rev() {
        let (previous, current) = (&window[0], &window[1]);
        let Some(preintegration) = current.preintegration else {
            break;
        };
        if preintegration.duration_seconds <= 0.01 {
            break;
        }
        pairs.push((
            current.position - previous.position,
            preintegration.duration_seconds,
            preintegration.delta_velocity,
            preintegration.delta_position,
        ));
    }
    pairs.reverse();
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("AR_DEBUG_SCALE").is_some() {
        let with_preint = map
            .keyframes
            .iter()
            .filter(|keyframe| keyframe.preintegration.is_some())
            .count();
        let span: f64 = pairs.iter().map(|(_, dt, _, _)| dt).sum();
        let excitation: f64 = pairs.iter().map(|(_, _, dv, _)| dv.length()).sum();
        eprintln!(
            "scale-chain pairs={} kf_with_preint={}/{} span={:.2}s excitation={:.2}",
            pairs.len(),
            with_preint,
            map.keyframes.len(),
            span,
            excitation
        );
    }
    if pairs.len() < MIN_PAIRS {
        return None;
    }
    let span: f64 = pairs.iter().map(|(_, dt, _, _)| dt).sum();
    let excitation: f64 = pairs.iter().map(|(_, _, dv, _)| dv.length()).sum();
    if span < MIN_SPAN_SECONDS || excitation < MIN_EXCITATION_MPS {
        return None;
    }

    // Per-pair unit-time variables: u_i = visual mean velocity, w_i = the
    // inertially measured mean velocity (Δp_i/dt_i plus the chained Δv
    // prefix). The kinematics give s·u_i − v0 = w_i; subtracting the
    // dt-weighted means eliminates v0 exactly, leaving pure proportionality
    // in the centered variables. The slope is then estimated symmetrically
    // (geometric mean of the forward and reverse regressions — total least
    // squares for a scalar slope): a one-sided regression on the noisy visual
    // displacements is biased toward zero (regression dilution), which showed
    // up against ARKit ground truth as ~2.5× underestimated ratios.
    let mut velocity_prefix = DVec3::ZERO;
    let mut unit: Vec<(f64, DVec3, DVec3)> = Vec::with_capacity(pairs.len());
    let mut weight_sum = 0.0;
    let mut mean_u = DVec3::ZERO;
    let mut mean_w = DVec3::ZERO;
    for (displacement, dt, delta_velocity, delta_position) in &pairs {
        let u = *displacement / *dt;
        let w = *delta_position / *dt + velocity_prefix;
        unit.push((*dt, u, w));
        weight_sum += dt;
        mean_u += u * *dt;
        mean_w += w * *dt;
        velocity_prefix += *delta_velocity;
    }
    if weight_sum <= 1.0e-9 {
        return None;
    }
    mean_u /= weight_sum;
    mean_w /= weight_sum;
    let mut uu = 0.0;
    let mut ww = 0.0;
    let mut uw = 0.0;
    for (dt, u, w) in &unit {
        let cu = *u - mean_u;
        let cw = *w - mean_w;
        uu += dt * cu.length_squared();
        ww += dt * cw.length_squared();
        uw += dt * cu.dot(cw);
    }
    if uu <= 1.0e-9 || ww <= 1.0e-9 {
        return None;
    }
    // Require genuine correlation before trusting the slope: uncorrelated
    // noise must not set the gauge.
    let correlation = uw / (uu.sqrt() * ww.sqrt());
    if correlation < 0.4 {
        return None;
    }
    let ratio = (ww / uu).sqrt();
    if !ratio.is_finite() || !(MIN_RATIO..=MAX_RATIO).contains(&ratio) {
        return None;
    }
    Some(ScaleEstimate {
        ratio,
        excitation,
        pairs: pairs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{Keyframe, Map, Preintegration};
    use vio_core::{DQuat, DVec3};

    fn keyframe_at(position: DVec3, preintegration: Option<Preintegration>) -> Keyframe {
        Keyframe {
            id: 0,
            position,
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            observations: Vec::new(),
            preintegration,
            luma: Vec::new(),
            luma_width: 0,
            luma_height: 0,
            descriptor: Vec::new(),
            full_luma: Vec::new(),
            full_width: 0,
            full_height: 0,
        }
    }

    #[test]
    fn recovers_known_scale_mismatch_under_excitation() {
        // True trajectory: 1D oscillation x(t) = 0.5·sin(1.5t); the map holds
        // it at 1/3 scale. Preintegration carries the true metric kinematics.
        let true_scale = 3.0;
        let dt = 0.4_f64;
        let position = |t: f64| DVec3::new(0.5 * (1.5 * t).sin(), 1.6, -0.2 * (1.1 * t).sin());
        let velocity = |t: f64| {
            DVec3::new(
                0.5 * 1.5 * (1.5 * t).cos(),
                0.0,
                -0.2 * 1.1 * (1.1 * t).cos(),
            )
        };

        let mut map = Map::new();
        for step in 0..10 {
            let t = f64::from(step) * dt;
            let preintegration = (step > 0).then(|| {
                let t_previous = f64::from(step - 1) * dt;
                Preintegration {
                    duration_seconds: dt,
                    delta_velocity: velocity(t) - velocity(t_previous),
                    delta_position: position(t)
                        - position(t_previous)
                        - velocity(t_previous) * dt,
                    sample_count: 24,
                }
            });
            map.push_keyframe(keyframe_at(position(t) / true_scale, preintegration));
        }

        let estimate = estimate_scale(&map).expect("scale should be observable");
        assert!(
            (estimate.ratio - true_scale).abs() < 0.15,
            "ratio={} expected ~{}",
            estimate.ratio,
            true_scale
        );
        assert!(estimate.excitation > MIN_EXCITATION_MPS);
    }

    #[test]
    fn refuses_smooth_motion_without_excitation() {
        // Constant velocity: scale is unobservable and must not be estimated.
        let mut map = Map::new();
        for step in 0..10 {
            let t = f64::from(step) * 0.4;
            let preintegration = (step > 0).then(|| Preintegration {
                duration_seconds: 0.4,
                delta_velocity: DVec3::ZERO,
                delta_position: DVec3::ZERO,
                sample_count: 24,
            });
            map.push_keyframe(keyframe_at(
                DVec3::new(0.3 * t, 1.6, 0.0),
                preintegration,
            ));
        }
        assert!(estimate_scale(&map).is_none());
    }
}
