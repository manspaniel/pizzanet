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
const MIN_RATIO: f64 = 0.15;
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
    if pairs.len() < MIN_PAIRS {
        return None;
    }
    let span: f64 = pairs.iter().map(|(_, dt, _, _)| dt).sum();
    let excitation: f64 = pairs.iter().map(|(_, _, dv, _)| dv.length()).sum();
    if span < MIN_SPAN_SECONDS || excitation < MIN_EXCITATION_MPS {
        return None;
    }

    // Least squares over x = [s, v0x, v0y, v0z]. For pair i, with the chained
    // velocity prefix P_i = Σ_{j<i} Δv_j:
    //   s·d_i − v0·dt_i = Δp_i + P_i·dt_i    (three rows)
    let mut normal = [[0.0_f64; 4]; 4];
    let mut rhs = [0.0_f64; 4];
    let mut velocity_prefix = DVec3::ZERO;
    for (displacement, dt, delta_velocity, delta_position) in &pairs {
        let target = *delta_position + velocity_prefix * *dt;
        for axis in 0..3 {
            // Row: [d_i[axis], −dt·e_axis] · x = target[axis]
            let mut row = [0.0_f64; 4];
            row[0] = displacement[axis];
            row[1 + axis] = -*dt;
            for column in 0..4 {
                for inner in 0..4 {
                    normal[column][inner] += row[column] * row[inner];
                }
                rhs[column] += row[column] * target[axis];
            }
        }
        velocity_prefix += *delta_velocity;
    }
    let solution = solve_4x4(normal, rhs)?;
    let ratio = solution[0];
    if !ratio.is_finite() || !(MIN_RATIO..=MAX_RATIO).contains(&ratio) {
        return None;
    }
    Some(ScaleEstimate {
        ratio,
        excitation,
        pairs: pairs.len(),
    })
}

/// Plain Gaussian elimination with partial pivoting for the 4×4 normal system.
fn solve_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for pivot in 0..4 {
        let mut best = pivot;
        for row in pivot + 1..4 {
            if a[row][pivot].abs() > a[best][pivot].abs() {
                best = row;
            }
        }
        if a[best][pivot].abs() < 1.0e-9 {
            return None;
        }
        a.swap(pivot, best);
        b.swap(pivot, best);
        for row in pivot + 1..4 {
            let factor = a[row][pivot] / a[pivot][pivot];
            let (upper, lower) = a.split_at_mut(row);
            for (column, value) in lower[0].iter_mut().enumerate().skip(pivot) {
                *value -= factor * upper[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    let mut x = [0.0_f64; 4];
    for row in (0..4).rev() {
        let mut sum = b[row];
        for column in row + 1..4 {
            sum -= a[row][column] * x[column];
        }
        x[row] = sum / a[row][row];
    }
    x.iter().all(|value| value.is_finite()).then_some(x)
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
