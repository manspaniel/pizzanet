use serde::{Deserialize, Serialize};

/// Building-relative side, with front at negative Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    /// Negative Z side.
    Front,
    /// Positive X side.
    Right,
    /// Positive Z side.
    Back,
    /// Negative X side.
    Left,
}

/// Broad semantic class used by part masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaceClass {
    /// Shallow outer roof skirt below the pitch break.
    LowerSkirt,
    /// Steep crown sides and the crown's flat top.
    UpperCrown,
    /// Closing face on the eave plane.
    Underside,
}

impl FaceClass {
    /// Every broad face class in stable mask order.
    pub const ALL: [Self; 3] = [Self::LowerSkirt, Self::UpperCrown, Self::Underside];

    /// Stable nonzero integer written to semantic-part masks.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        match self {
            Self::LowerSkirt => 1,
            Self::UpperCrown => 2,
            Self::Underside => 3,
        }
    }

    /// Stable manifest name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowerSkirt => "lower_skirt",
            Self::UpperCrown => "upper_crown",
            Self::Underside => "underside",
        }
    }
}

/// Stable semantic identifier for a planar roof face.
///
/// Values are explicitly assigned because they are also written to integer GPU
/// targets in synthetic datasets. They must not be renumbered within a schema
/// version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[serde(rename_all = "snake_case")]
pub enum FaceId {
    /// Lower front slope.
    LowerFront = 0x0100,
    /// Lower right slope.
    LowerRight = 0x0101,
    /// Lower back slope.
    LowerBack = 0x0102,
    /// Lower left slope.
    LowerLeft = 0x0103,
    /// Steep front face of the crown frustum.
    UpperFront = 0x0200,
    /// Steep right face of the crown frustum.
    UpperRight = 0x0201,
    /// Steep back face of the crown frustum.
    UpperBack = 0x0202,
    /// Steep left face of the crown frustum.
    UpperLeft = 0x0203,
    /// Flat rectangular crown top.
    CrownTop = 0x0204,
    /// Bottom closing face.
    Underside = 0x0300,
}

impl FaceId {
    /// Every face in deterministic mesh order.
    pub const ALL: [Self; 10] = [
        Self::LowerFront,
        Self::LowerRight,
        Self::LowerBack,
        Self::LowerLeft,
        Self::UpperFront,
        Self::UpperRight,
        Self::UpperBack,
        Self::UpperLeft,
        Self::CrownTop,
        Self::Underside,
    ];

    /// Stable numeric ID suitable for an integer render target.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Decodes a stable integer render-target value.
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x0100 => Some(Self::LowerFront),
            0x0101 => Some(Self::LowerRight),
            0x0102 => Some(Self::LowerBack),
            0x0103 => Some(Self::LowerLeft),
            0x0200 => Some(Self::UpperFront),
            0x0201 => Some(Self::UpperRight),
            0x0202 => Some(Self::UpperBack),
            0x0203 => Some(Self::UpperLeft),
            0x0204 => Some(Self::CrownTop),
            0x0300 => Some(Self::Underside),
            _ => None,
        }
    }

    /// Stable manifest name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowerFront => "lower_front",
            Self::LowerRight => "lower_right",
            Self::LowerBack => "lower_back",
            Self::LowerLeft => "lower_left",
            Self::UpperFront => "upper_front",
            Self::UpperRight => "upper_right",
            Self::UpperBack => "upper_back",
            Self::UpperLeft => "upper_left",
            Self::CrownTop => "crown_top",
            Self::Underside => "underside",
        }
    }

    /// Broad semantic part class.
    #[must_use]
    pub const fn class(self) -> FaceClass {
        match self {
            Self::LowerFront | Self::LowerRight | Self::LowerBack | Self::LowerLeft => {
                FaceClass::LowerSkirt
            }
            Self::UpperFront
            | Self::UpperRight
            | Self::UpperBack
            | Self::UpperLeft
            | Self::CrownTop => FaceClass::UpperCrown,
            Self::Underside => FaceClass::Underside,
        }
    }

    /// Building side for side faces, or `None` for horizontal faces.
    #[must_use]
    pub const fn side(self) -> Option<Side> {
        match self {
            Self::LowerFront | Self::UpperFront => Some(Side::Front),
            Self::LowerRight | Self::UpperRight => Some(Side::Right),
            Self::LowerBack | Self::UpperBack => Some(Side::Back),
            Self::LowerLeft | Self::UpperLeft => Some(Side::Left),
            Self::CrownTop | Self::Underside => None,
        }
    }
}

/// Structural landmark category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeypointCategory {
    /// Corner of the outer eave rectangle.
    EaveCorner,
    /// Corner of the pitch-break/crown-base rectangle.
    ShoulderCorner,
    /// Corner of the flat crown-top rectangle.
    CrownTopCorner,
}

/// Stable identifier and serialized name for a structural 3D keypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[serde(rename_all = "snake_case")]
pub enum KeypointId {
    /// Front-left eave corner.
    EaveFrontLeft = 0x0100,
    /// Front-right eave corner.
    EaveFrontRight = 0x0101,
    /// Back-right eave corner.
    EaveBackRight = 0x0102,
    /// Back-left eave corner.
    EaveBackLeft = 0x0103,
    /// Front-left shoulder/crown-base corner.
    ShoulderFrontLeft = 0x0200,
    /// Front-right shoulder/crown-base corner.
    ShoulderFrontRight = 0x0201,
    /// Back-right shoulder/crown-base corner.
    ShoulderBackRight = 0x0202,
    /// Back-left shoulder/crown-base corner.
    ShoulderBackLeft = 0x0203,
    /// Front-left corner of the crown top.
    CrownTopFrontLeft = 0x0300,
    /// Front-right corner of the crown top.
    CrownTopFrontRight = 0x0301,
    /// Back-right corner of the crown top.
    CrownTopBackRight = 0x0302,
    /// Back-left corner of the crown top.
    CrownTopBackLeft = 0x0303,
}

impl KeypointId {
    /// Every keypoint in deterministic annotation order.
    pub const ALL: [Self; 12] = [
        Self::EaveFrontLeft,
        Self::EaveFrontRight,
        Self::EaveBackRight,
        Self::EaveBackLeft,
        Self::ShoulderFrontLeft,
        Self::ShoulderFrontRight,
        Self::ShoulderBackRight,
        Self::ShoulderBackLeft,
        Self::CrownTopFrontLeft,
        Self::CrownTopFrontRight,
        Self::CrownTopBackRight,
        Self::CrownTopBackLeft,
    ];

    /// Stable numeric identifier.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Stable human-readable landmark name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EaveFrontLeft => "eave_front_left",
            Self::EaveFrontRight => "eave_front_right",
            Self::EaveBackRight => "eave_back_right",
            Self::EaveBackLeft => "eave_back_left",
            Self::ShoulderFrontLeft => "shoulder_front_left",
            Self::ShoulderFrontRight => "shoulder_front_right",
            Self::ShoulderBackRight => "shoulder_back_right",
            Self::ShoulderBackLeft => "shoulder_back_left",
            Self::CrownTopFrontLeft => "crown_top_front_left",
            Self::CrownTopFrontRight => "crown_top_front_right",
            Self::CrownTopBackRight => "crown_top_back_right",
            Self::CrownTopBackLeft => "crown_top_back_left",
        }
    }

    /// Semantic landmark category.
    #[must_use]
    pub const fn category(self) -> KeypointCategory {
        match self {
            Self::EaveFrontLeft
            | Self::EaveFrontRight
            | Self::EaveBackRight
            | Self::EaveBackLeft => KeypointCategory::EaveCorner,
            Self::ShoulderFrontLeft
            | Self::ShoulderFrontRight
            | Self::ShoulderBackRight
            | Self::ShoulderBackLeft => KeypointCategory::ShoulderCorner,
            Self::CrownTopFrontLeft
            | Self::CrownTopFrontRight
            | Self::CrownTopBackRight
            | Self::CrownTopBackLeft => KeypointCategory::CrownTopCorner,
        }
    }
}

/// Structural line category for projected edge supervision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeCategory {
    /// Outer roof boundary.
    Eave,
    /// Break between shallow skirt and steep crown.
    ShoulderBreak,
    /// Corner line across the lower skirt.
    LowerHip,
    /// Corner line across the upper crown.
    UpperHip,
    /// Perimeter of the flat crown top.
    CrownTopPerimeter,
}

/// Stable identifier and name for a semantic structural edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[serde(rename_all = "snake_case")]
pub enum EdgeId {
    /// Front eave boundary.
    EaveFront = 0x0100,
    /// Right eave boundary.
    EaveRight = 0x0101,
    /// Back eave boundary.
    EaveBack = 0x0102,
    /// Left eave boundary.
    EaveLeft = 0x0103,
    /// Front pitch-break edge.
    ShoulderFront = 0x0200,
    /// Right pitch-break edge.
    ShoulderRight = 0x0201,
    /// Back pitch-break edge.
    ShoulderBack = 0x0202,
    /// Left pitch-break edge.
    ShoulderLeft = 0x0203,
    /// Lower front-left hip.
    LowerHipFrontLeft = 0x0300,
    /// Lower front-right hip.
    LowerHipFrontRight = 0x0301,
    /// Lower back-right hip.
    LowerHipBackRight = 0x0302,
    /// Lower back-left hip.
    LowerHipBackLeft = 0x0303,
    /// Upper front-left hip.
    UpperHipFrontLeft = 0x0400,
    /// Upper front-right hip.
    UpperHipFrontRight = 0x0401,
    /// Upper back-right hip.
    UpperHipBackRight = 0x0402,
    /// Upper back-left hip.
    UpperHipBackLeft = 0x0403,
    /// Front edge of the flat crown top.
    CrownTopFront = 0x0500,
    /// Right edge of the flat crown top.
    CrownTopRight = 0x0501,
    /// Back edge of the flat crown top.
    CrownTopBack = 0x0502,
    /// Left edge of the flat crown top.
    CrownTopLeft = 0x0503,
}

impl EdgeId {
    /// Every edge in deterministic annotation order.
    pub const ALL: [Self; 20] = [
        Self::EaveFront,
        Self::EaveRight,
        Self::EaveBack,
        Self::EaveLeft,
        Self::ShoulderFront,
        Self::ShoulderRight,
        Self::ShoulderBack,
        Self::ShoulderLeft,
        Self::LowerHipFrontLeft,
        Self::LowerHipFrontRight,
        Self::LowerHipBackRight,
        Self::LowerHipBackLeft,
        Self::UpperHipFrontLeft,
        Self::UpperHipFrontRight,
        Self::UpperHipBackRight,
        Self::UpperHipBackLeft,
        Self::CrownTopFront,
        Self::CrownTopRight,
        Self::CrownTopBack,
        Self::CrownTopLeft,
    ];

    /// Stable numeric identifier.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Semantic line category.
    #[must_use]
    pub const fn category(self) -> EdgeCategory {
        match self {
            Self::EaveFront | Self::EaveRight | Self::EaveBack | Self::EaveLeft => {
                EdgeCategory::Eave
            }
            Self::ShoulderFront | Self::ShoulderRight | Self::ShoulderBack | Self::ShoulderLeft => {
                EdgeCategory::ShoulderBreak
            }
            Self::LowerHipFrontLeft
            | Self::LowerHipFrontRight
            | Self::LowerHipBackRight
            | Self::LowerHipBackLeft => EdgeCategory::LowerHip,
            Self::UpperHipFrontLeft
            | Self::UpperHipFrontRight
            | Self::UpperHipBackRight
            | Self::UpperHipBackLeft => EdgeCategory::UpperHip,
            Self::CrownTopFront | Self::CrownTopRight | Self::CrownTopBack | Self::CrownTopLeft => {
                EdgeCategory::CrownTopPerimeter
            }
        }
    }

    /// Stable keypoint endpoints for this edge.
    #[must_use]
    pub const fn endpoints(self) -> (KeypointId, KeypointId) {
        use KeypointId::{
            CrownTopBackLeft, CrownTopBackRight, CrownTopFrontLeft, CrownTopFrontRight,
            EaveBackLeft, EaveBackRight, EaveFrontLeft, EaveFrontRight, ShoulderBackLeft,
            ShoulderBackRight, ShoulderFrontLeft, ShoulderFrontRight,
        };

        match self {
            Self::EaveFront => (EaveFrontLeft, EaveFrontRight),
            Self::EaveRight => (EaveFrontRight, EaveBackRight),
            Self::EaveBack => (EaveBackRight, EaveBackLeft),
            Self::EaveLeft => (EaveBackLeft, EaveFrontLeft),
            Self::ShoulderFront => (ShoulderFrontLeft, ShoulderFrontRight),
            Self::ShoulderRight => (ShoulderFrontRight, ShoulderBackRight),
            Self::ShoulderBack => (ShoulderBackRight, ShoulderBackLeft),
            Self::ShoulderLeft => (ShoulderBackLeft, ShoulderFrontLeft),
            Self::LowerHipFrontLeft => (EaveFrontLeft, ShoulderFrontLeft),
            Self::LowerHipFrontRight => (EaveFrontRight, ShoulderFrontRight),
            Self::LowerHipBackRight => (EaveBackRight, ShoulderBackRight),
            Self::LowerHipBackLeft => (EaveBackLeft, ShoulderBackLeft),
            Self::UpperHipFrontLeft => (ShoulderFrontLeft, CrownTopFrontLeft),
            Self::UpperHipFrontRight => (ShoulderFrontRight, CrownTopFrontRight),
            Self::UpperHipBackRight => (ShoulderBackRight, CrownTopBackRight),
            Self::UpperHipBackLeft => (ShoulderBackLeft, CrownTopBackLeft),
            Self::CrownTopFront => (CrownTopFrontLeft, CrownTopFrontRight),
            Self::CrownTopRight => (CrownTopFrontRight, CrownTopBackRight),
            Self::CrownTopBack => (CrownTopBackRight, CrownTopBackLeft),
            Self::CrownTopLeft => (CrownTopBackLeft, CrownTopFrontLeft),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_semantic_ids_are_stable_for_schema_two() {
        assert_eq!(FaceId::LowerFront.as_u32(), 0x0100);
        assert_eq!(FaceId::UpperLeft.as_u32(), 0x0203);
        assert_eq!(FaceId::CrownTop.as_u32(), 0x0204);
        assert_eq!(FaceId::Underside.as_u32(), 0x0300);
        assert_eq!(FaceId::from_u32(0x0204), Some(FaceId::CrownTop));
        assert_eq!(FaceId::from_u32(0), None);
        assert_eq!(FaceClass::UpperCrown.as_u16(), 2);
        assert_eq!(KeypointId::CrownTopBackRight.as_u32(), 0x0302);
        assert_eq!(EdgeId::CrownTopLeft.as_u32(), 0x0503);
    }

    #[test]
    fn serialized_semantic_names_are_stable() {
        assert_eq!(
            serde_json::to_string(&FaceId::CrownTop).unwrap(),
            r#""crown_top""#
        );
        assert_eq!(
            serde_json::to_string(&KeypointId::CrownTopFrontLeft).unwrap(),
            r#""crown_top_front_left""#
        );
        assert_eq!(
            serde_json::to_string(&EdgeId::CrownTopBack).unwrap(),
            r#""crown_top_back""#
        );
    }

    #[test]
    fn categories_and_edge_endpoints_are_stable() {
        assert_eq!(FaceId::LowerRight.class(), FaceClass::LowerSkirt);
        assert_eq!(FaceId::CrownTop.class(), FaceClass::UpperCrown);
        assert_eq!(FaceId::UpperBack.side(), Some(Side::Back));
        assert_eq!(FaceId::CrownTop.side(), None);
        assert_eq!(FaceId::Underside.side(), None);
        assert_eq!(
            EdgeId::CrownTopFront.category(),
            EdgeCategory::CrownTopPerimeter
        );
        assert_eq!(
            EdgeId::UpperHipBackLeft.endpoints(),
            (KeypointId::ShoulderBackLeft, KeypointId::CrownTopBackLeft)
        );
    }
}
