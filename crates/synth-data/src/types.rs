use serde::{Deserialize, Serialize};

/// Inclusive floating-point sampling range.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FloatRange {
    /// Smallest permitted value.
    pub min: f32,
    /// Largest permitted value.
    pub max: f32,
}

impl FloatRange {
    /// Creates an inclusive range.
    #[must_use]
    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Returns whether both limits are finite and ordered.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min <= self.max
    }

    /// Returns the inclusive overlap of two valid ranges, if one exists.
    #[must_use]
    pub fn intersection(self, other: Self) -> Option<Self> {
        if !self.is_valid() || !other.is_valid() {
            return None;
        }
        let intersection = Self::new(self.min.max(other.min), self.max.min(other.max));
        intersection.is_valid().then_some(intersection)
    }

    /// Returns whether a finite value lies inside the inclusive range.
    #[must_use]
    pub fn contains(self, value: f32) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }
}

/// Inclusive integer sampling range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct U32Range {
    /// Smallest permitted value.
    pub min: u32,
    /// Largest permitted value.
    pub max: u32,
}

impl U32Range {
    /// Creates an inclusive range.
    #[must_use]
    pub const fn new(min: u32, max: u32) -> Self {
        Self { min, max }
    }

    /// Returns whether the limits are ordered.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.min <= self.max
    }
}

/// Two-dimensional value, normally expressed in pixels or normalized image space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f32,
    /// Vertical component.
    pub y: f32,
}

impl Vec2 {
    /// Creates a vector.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns whether both components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// Three-dimensional value in the manifest's declared coordinate system.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Creates a vector.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns whether all components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Rigid transform whose quaternion is stored in `(x, y, z, w)` order.
///
/// `translation` and `rotation_xyzw` describe a child-to-parent transform. For
/// example, `world_from_camera` maps camera-local points into world space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidTransform {
    /// Translation in parent coordinates.
    pub translation: Vec3,
    /// Unit quaternion in `(x, y, z, w)` order.
    pub rotation_xyzw: [f32; 4],
}

impl RigidTransform {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        translation: Vec3::new(0.0, 0.0, 0.0),
        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
    };

    /// Returns whether the components are finite and the quaternion is unit length.
    #[must_use]
    pub fn is_valid(self) -> bool {
        if !self.translation.is_finite() || self.rotation_xyzw.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let norm_squared = self
            .rotation_xyzw
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        (norm_squared - 1.0).abs() <= 1.0e-3
    }
}
