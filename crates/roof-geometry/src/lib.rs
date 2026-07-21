//! Shared parametric geometry for the classic flat-topped Pizza Hut roof.
//!
//! The crate deliberately contains no renderer, platform API, or native-only
//! dependency. Native synthetic-data tools and a later `wasm32` application can
//! therefore generate exactly the same mesh, semantic faces, landmarks, and
//! structural edges.
//!
//! Coordinates are right-handed: X runs left-to-right across the building, Y
//! points up, and the building front faces negative Z. The roof's eave plane is
//! Y = 0. Parameter values are scale-agnostic, although metres are convenient.

#![forbid(unsafe_code)]

mod geometry;
mod parameters;
mod semantics;

pub use geometry::{
    IndexedMesh, MeshFace, RoofGeometry, StructuralEdge, StructuralKeypoint, Vertex, generate_roof,
};
pub use parameters::{ParameterError, ParameterField, RoofParameters};
pub use semantics::{EdgeCategory, EdgeId, FaceClass, FaceId, KeypointCategory, KeypointId, Side};

/// Version of the serialized geometry contract and numeric semantic IDs.
///
/// Increment this only for an intentionally incompatible output change.
pub const GEOMETRY_SCHEMA_VERSION: u32 = 2;
