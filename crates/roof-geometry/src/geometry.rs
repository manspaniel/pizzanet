use serde::{Deserialize, Serialize};

use crate::{
    EdgeCategory, EdgeId, FaceClass, FaceId, GEOMETRY_SCHEMA_VERSION, KeypointCategory, KeypointId,
    ParameterError, RoofParameters,
};

/// A vertex with flat-shaded normal and coordinates local to its semantic face.
///
/// Vertices are intentionally duplicated at face boundaries so flat normals and
/// discontinuous face coordinates remain unambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    /// Position in roof-local XYZ coordinates.
    pub position: [f32; 3],
    /// Unit outward face normal.
    pub normal: [f32; 3],
    /// Normalized coordinates within the owning face.
    pub face_coord: [f32; 2],
    /// Stable semantic face identifier.
    pub face_id: FaceId,
}

/// Contiguous index range belonging to one semantic face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshFace {
    /// Stable face identifier.
    pub id: FaceId,
    /// Broad part-mask class.
    pub class: FaceClass,
    /// Offset into [`IndexedMesh::indices`].
    pub first_index: u32,
    /// Number of indices; always a multiple of three.
    pub index_count: u32,
}

/// Flat-shaded indexed triangle mesh grouped by stable semantic face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexedMesh {
    /// Face-local vertices.
    pub vertices: Vec<Vertex>,
    /// Counter-clockwise triangle indices when viewed from outside.
    pub indices: Vec<u32>,
    /// Semantic face spans in deterministic order.
    pub faces: Vec<MeshFace>,
}

/// A named semantic landmark positioned in roof-local 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralKeypoint {
    /// Stable keypoint identifier and serialized name.
    pub id: KeypointId,
    /// Landmark category.
    pub category: KeypointCategory,
    /// Roof-local XYZ position.
    pub position: [f32; 3],
}

impl StructuralKeypoint {
    /// Stable human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.id.as_str()
    }
}

/// A semantic line segment connecting two structural keypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralEdge {
    /// Stable edge identifier and serialized name.
    pub id: EdgeId,
    /// Edge category used for structural-edge targets.
    pub category: EdgeCategory,
    /// First endpoint.
    pub start: KeypointId,
    /// Second endpoint.
    pub end: KeypointId,
}

/// Complete deterministic geometry and annotations for one roof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoofGeometry {
    /// Serialization and semantic-contract version.
    pub schema_version: u32,
    /// Validated parameters that generated this geometry.
    pub parameters: RoofParameters,
    /// Renderable semantic triangle mesh.
    pub mesh: IndexedMesh,
    /// Named 3D structural landmarks.
    pub keypoints: Vec<StructuralKeypoint>,
    /// Semantic structural edges expressed through keypoint IDs.
    pub edges: Vec<StructuralEdge>,
}

impl RoofGeometry {
    /// Looks up a mesh face by stable ID.
    #[must_use]
    pub fn face(&self, id: FaceId) -> Option<&MeshFace> {
        self.mesh.faces.iter().find(|face| face.id == id)
    }

    /// Looks up a structural keypoint by stable ID.
    #[must_use]
    pub fn keypoint(&self, id: KeypointId) -> Option<&StructuralKeypoint> {
        self.keypoints.iter().find(|keypoint| keypoint.id == id)
    }

    /// Looks up a semantic edge by stable ID.
    #[must_use]
    pub fn edge(&self, id: EdgeId) -> Option<&StructuralEdge> {
        self.edges.iter().find(|edge| edge.id == id)
    }

    /// Resolves an edge to its two roof-local endpoint positions.
    #[must_use]
    pub fn edge_positions(&self, id: EdgeId) -> Option<([f32; 3], [f32; 3])> {
        let edge = self.edge(id)?;
        Some((
            self.keypoint(edge.start)?.position,
            self.keypoint(edge.end)?.position,
        ))
    }
}

/// Validates parameters and generates the classic flat-topped two-pitch roof.
pub fn generate_roof(parameters: &RoofParameters) -> Result<RoofGeometry, ParameterError> {
    parameters.validate()?;

    let points = ControlPoints::new(parameters);
    let mut mesh = MeshBuilder::default();
    let quad_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    mesh.push_face(
        FaceId::LowerFront,
        &[
            points.eave_fr,
            points.eave_fl,
            points.shoulder_fl,
            points.shoulder_fr,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::LowerRight,
        &[
            points.eave_br,
            points.eave_fr,
            points.shoulder_fr,
            points.shoulder_br,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::LowerBack,
        &[
            points.eave_bl,
            points.eave_br,
            points.shoulder_br,
            points.shoulder_bl,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::LowerLeft,
        &[
            points.eave_fl,
            points.eave_bl,
            points.shoulder_bl,
            points.shoulder_fl,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::UpperFront,
        &[
            points.shoulder_fr,
            points.shoulder_fl,
            points.crown_top_fl,
            points.crown_top_fr,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::UpperRight,
        &[
            points.shoulder_br,
            points.shoulder_fr,
            points.crown_top_fr,
            points.crown_top_br,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::UpperBack,
        &[
            points.shoulder_bl,
            points.shoulder_br,
            points.crown_top_br,
            points.crown_top_bl,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::UpperLeft,
        &[
            points.shoulder_fl,
            points.shoulder_bl,
            points.crown_top_bl,
            points.crown_top_fl,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::CrownTop,
        &[
            points.crown_top_fl,
            points.crown_top_bl,
            points.crown_top_br,
            points.crown_top_fr,
        ],
        &quad_coords,
    );
    mesh.push_face(
        FaceId::Underside,
        &[
            points.eave_fl,
            points.eave_fr,
            points.eave_br,
            points.eave_bl,
        ],
        &quad_coords,
    );

    let keypoints = KeypointId::ALL
        .into_iter()
        .map(|id| StructuralKeypoint {
            id,
            category: id.category(),
            position: points.keypoint(id),
        })
        .collect();

    let edges = EdgeId::ALL
        .into_iter()
        .map(|id| {
            let (start, end) = id.endpoints();
            StructuralEdge {
                id,
                category: id.category(),
                start,
                end,
            }
        })
        .collect();

    Ok(RoofGeometry {
        schema_version: GEOMETRY_SCHEMA_VERSION,
        parameters: *parameters,
        mesh: mesh.finish(),
        keypoints,
        edges,
    })
}

struct ControlPoints {
    eave_fl: [f32; 3],
    eave_fr: [f32; 3],
    eave_br: [f32; 3],
    eave_bl: [f32; 3],
    shoulder_fl: [f32; 3],
    shoulder_fr: [f32; 3],
    shoulder_br: [f32; 3],
    shoulder_bl: [f32; 3],
    crown_top_fl: [f32; 3],
    crown_top_fr: [f32; 3],
    crown_top_br: [f32; 3],
    crown_top_bl: [f32; 3],
}

impl ControlPoints {
    fn new(parameters: &RoofParameters) -> Self {
        let eave_x = parameters.eave_width * 0.5;
        let eave_z = parameters.eave_depth * 0.5;
        let shoulder_x = parameters.shoulder_width * 0.5;
        let shoulder_z = parameters.shoulder_depth * 0.5;
        let crown_top_x = parameters.crown_top_width * 0.5;
        let crown_top_z = parameters.crown_top_depth * 0.5;
        let shoulder_y = parameters.lower_rise;
        let crown_top_y = parameters.lower_rise + parameters.upper_rise;

        Self {
            eave_fl: [-eave_x, 0.0, -eave_z],
            eave_fr: [eave_x, 0.0, -eave_z],
            eave_br: [eave_x, 0.0, eave_z],
            eave_bl: [-eave_x, 0.0, eave_z],
            shoulder_fl: [-shoulder_x, shoulder_y, -shoulder_z],
            shoulder_fr: [shoulder_x, shoulder_y, -shoulder_z],
            shoulder_br: [shoulder_x, shoulder_y, shoulder_z],
            shoulder_bl: [-shoulder_x, shoulder_y, shoulder_z],
            crown_top_fl: [-crown_top_x, crown_top_y, -crown_top_z],
            crown_top_fr: [crown_top_x, crown_top_y, -crown_top_z],
            crown_top_br: [crown_top_x, crown_top_y, crown_top_z],
            crown_top_bl: [-crown_top_x, crown_top_y, crown_top_z],
        }
    }

    fn keypoint(&self, id: KeypointId) -> [f32; 3] {
        match id {
            KeypointId::EaveFrontLeft => self.eave_fl,
            KeypointId::EaveFrontRight => self.eave_fr,
            KeypointId::EaveBackRight => self.eave_br,
            KeypointId::EaveBackLeft => self.eave_bl,
            KeypointId::ShoulderFrontLeft => self.shoulder_fl,
            KeypointId::ShoulderFrontRight => self.shoulder_fr,
            KeypointId::ShoulderBackRight => self.shoulder_br,
            KeypointId::ShoulderBackLeft => self.shoulder_bl,
            KeypointId::CrownTopFrontLeft => self.crown_top_fl,
            KeypointId::CrownTopFrontRight => self.crown_top_fr,
            KeypointId::CrownTopBackRight => self.crown_top_br,
            KeypointId::CrownTopBackLeft => self.crown_top_bl,
        }
    }
}

#[derive(Default)]
struct MeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    faces: Vec<MeshFace>,
}

impl MeshBuilder {
    fn push_face(&mut self, id: FaceId, positions: &[[f32; 3]], coords: &[[f32; 2]]) {
        debug_assert!(positions.len() >= 3);
        debug_assert_eq!(positions.len(), coords.len());

        let first_vertex = u32::try_from(self.vertices.len()).expect("roof mesh fits in u32");
        let first_index = u32::try_from(self.indices.len()).expect("roof mesh fits in u32");
        let normal = normalized(cross(
            subtract(positions[1], positions[0]),
            subtract(positions[2], positions[0]),
        ));

        self.vertices.extend(
            positions
                .iter()
                .zip(coords)
                .map(|(&position, &face_coord)| Vertex {
                    position,
                    normal,
                    face_coord,
                    face_id: id,
                }),
        );

        for offset in 1..positions.len() - 1 {
            let offset = u32::try_from(offset).expect("roof face fits in u32");
            self.indices.extend_from_slice(&[
                first_vertex,
                first_vertex + offset,
                first_vertex + offset + 1,
            ]);
        }

        let index_count =
            u32::try_from(self.indices.len()).expect("roof mesh fits in u32") - first_index;
        self.faces.push(MeshFace {
            id,
            class: id.class(),
            first_index,
            index_count,
        });
    }

    fn finish(self) -> IndexedMesh {
        IndexedMesh {
            vertices: self.vertices,
            indices: self.indices,
            faces: self.faces,
        }
    }
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(vector: [f32; 3]) -> [f32; 3] {
    let length = vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    debug_assert!(length.is_finite() && length > 0.0);
    [vector[0] / length, vector[1] / length, vector[2] / length]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn geometry() -> RoofGeometry {
        generate_roof(&RoofParameters::default()).unwrap()
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn length(vector: [f32; 3]) -> f32 {
        dot(vector, vector).sqrt()
    }

    fn assert_mirrored_x(left: KeypointId, right: KeypointId, roof: &RoofGeometry) {
        let left = roof.keypoint(left).unwrap().position;
        let right = roof.keypoint(right).unwrap().position;
        assert!((left[0] + right[0]).abs() < EPSILON);
        assert!((left[1] - right[1]).abs() < EPSILON);
        assert!((left[2] - right[2]).abs() < EPSILON);
    }

    fn assert_mirrored_z(front: KeypointId, back: KeypointId, roof: &RoofGeometry) {
        let front = roof.keypoint(front).unwrap().position;
        let back = roof.keypoint(back).unwrap().position;
        assert!((front[0] - back[0]).abs() < EPSILON);
        assert!((front[1] - back[1]).abs() < EPSILON);
        assert!((front[2] + back[2]).abs() < EPSILON);
    }

    #[test]
    fn output_is_finite_and_face_coordinates_are_normalized() {
        let roof = geometry();
        for vertex in &roof.mesh.vertices {
            assert!(vertex.position.into_iter().all(f32::is_finite));
            assert!(vertex.normal.into_iter().all(f32::is_finite));
            assert!(vertex.face_coord.into_iter().all(f32::is_finite));
            assert!(
                vertex
                    .face_coord
                    .into_iter()
                    .all(|coordinate| (0.0..=1.0).contains(&coordinate))
            );
        }
        for keypoint in &roof.keypoints {
            assert!(keypoint.position.into_iter().all(f32::is_finite));
        }
    }

    #[test]
    fn all_indices_and_face_spans_are_in_bounds() {
        let roof = geometry();
        assert!(
            roof.mesh
                .indices
                .iter()
                .all(|&index| index < roof.mesh.vertices.len() as u32)
        );

        let mut expected_first = 0;
        for face in &roof.mesh.faces {
            assert_eq!(face.first_index, expected_first);
            assert_eq!(face.index_count % 3, 0);
            expected_first += face.index_count;
            assert!(expected_first <= roof.mesh.indices.len() as u32);
        }
        assert_eq!(expected_first, roof.mesh.indices.len() as u32);
    }

    #[test]
    fn triangle_winding_matches_unit_vertex_normals() {
        let roof = geometry();
        for face in &roof.mesh.faces {
            let start = face.first_index as usize;
            let end = start + face.index_count as usize;
            for triangle in roof.mesh.indices[start..end].chunks_exact(3) {
                let a = roof.mesh.vertices[triangle[0] as usize];
                let b = roof.mesh.vertices[triangle[1] as usize];
                let c = roof.mesh.vertices[triangle[2] as usize];
                let triangle_normal = normalized(cross(
                    subtract(b.position, a.position),
                    subtract(c.position, a.position),
                ));
                assert_eq!(a.face_id, face.id);
                assert_eq!(b.face_id, face.id);
                assert_eq!(c.face_id, face.id);
                assert!((length(a.normal) - 1.0).abs() < EPSILON);
                assert!(dot(triangle_normal, a.normal) > 1.0 - EPSILON);
                assert_eq!(a.normal, b.normal);
                assert_eq!(a.normal, c.normal);
            }
        }
    }

    #[test]
    fn visible_normals_point_up_and_outward() {
        let roof = geometry();
        let expected = [
            (FaceId::LowerFront, [0.0, 0.0, -1.0]),
            (FaceId::LowerRight, [1.0, 0.0, 0.0]),
            (FaceId::LowerBack, [0.0, 0.0, 1.0]),
            (FaceId::LowerLeft, [-1.0, 0.0, 0.0]),
            (FaceId::UpperFront, [0.0, 0.0, -1.0]),
            (FaceId::UpperRight, [1.0, 0.0, 0.0]),
            (FaceId::UpperBack, [0.0, 0.0, 1.0]),
            (FaceId::UpperLeft, [-1.0, 0.0, 0.0]),
        ];

        for (id, outward) in expected {
            let face = roof.face(id).unwrap();
            let vertex = roof.mesh.vertices[roof.mesh.indices[face.first_index as usize] as usize];
            assert!(vertex.normal[1] > 0.0);
            assert!(dot(vertex.normal, outward) > 0.0);
        }

        let crown_top = roof.face(FaceId::CrownTop).unwrap();
        let normal =
            roof.mesh.vertices[roof.mesh.indices[crown_top.first_index as usize] as usize].normal;
        assert!(normal[1] > 1.0 - EPSILON);

        let underside = roof.face(FaceId::Underside).unwrap();
        let normal =
            roof.mesh.vertices[roof.mesh.indices[underside.first_index as usize] as usize].normal;
        assert!(normal[1] < -1.0 + EPSILON);
    }

    #[test]
    fn semantic_collections_are_complete_and_ordered() {
        let roof = geometry();
        assert_eq!(roof.mesh.faces.len(), 10);
        assert_eq!(roof.mesh.vertices.len(), 40);
        assert_eq!(roof.mesh.indices.len(), 60);
        assert!(roof.mesh.faces.iter().all(|face| face.index_count == 6));
        assert_eq!(
            roof.mesh
                .faces
                .iter()
                .map(|face| face.id)
                .collect::<Vec<_>>(),
            FaceId::ALL
        );
        assert_eq!(
            roof.keypoints
                .iter()
                .map(|keypoint| keypoint.id)
                .collect::<Vec<_>>(),
            KeypointId::ALL
        );
        assert_eq!(roof.keypoints.len(), 12);
        assert_eq!(
            roof.edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
            EdgeId::ALL
        );
        assert_eq!(roof.edges.len(), 20);
        for edge in &roof.edges {
            assert_eq!((edge.start, edge.end), edge.id.endpoints());
            assert!(roof.edge_positions(edge.id).is_some());
        }
    }

    #[test]
    fn keypoints_are_bilaterally_symmetric() {
        let roof = geometry();
        assert_mirrored_x(KeypointId::EaveFrontLeft, KeypointId::EaveFrontRight, &roof);
        assert_mirrored_x(KeypointId::EaveBackLeft, KeypointId::EaveBackRight, &roof);
        assert_mirrored_x(
            KeypointId::ShoulderFrontLeft,
            KeypointId::ShoulderFrontRight,
            &roof,
        );
        assert_mirrored_x(
            KeypointId::ShoulderBackLeft,
            KeypointId::ShoulderBackRight,
            &roof,
        );
        assert_mirrored_x(
            KeypointId::CrownTopFrontLeft,
            KeypointId::CrownTopFrontRight,
            &roof,
        );
        assert_mirrored_x(
            KeypointId::CrownTopBackLeft,
            KeypointId::CrownTopBackRight,
            &roof,
        );

        assert_mirrored_z(KeypointId::EaveFrontLeft, KeypointId::EaveBackLeft, &roof);
        assert_mirrored_z(KeypointId::EaveFrontRight, KeypointId::EaveBackRight, &roof);
        assert_mirrored_z(
            KeypointId::ShoulderFrontLeft,
            KeypointId::ShoulderBackLeft,
            &roof,
        );
        assert_mirrored_z(
            KeypointId::ShoulderFrontRight,
            KeypointId::ShoulderBackRight,
            &roof,
        );
        assert_mirrored_z(
            KeypointId::CrownTopFrontLeft,
            KeypointId::CrownTopBackLeft,
            &roof,
        );
        assert_mirrored_z(
            KeypointId::CrownTopFrontRight,
            KeypointId::CrownTopBackRight,
            &roof,
        );
    }

    #[test]
    fn crown_top_has_requested_dimensions_and_height() {
        let parameters = RoofParameters {
            crown_top_width: 10.0,
            crown_top_depth: 5.0,
            lower_rise: 3.0,
            upper_rise: 4.0,
            ..RoofParameters::default()
        };
        let roof = generate_roof(&parameters).unwrap();
        let front_left = roof
            .keypoint(KeypointId::CrownTopFrontLeft)
            .unwrap()
            .position;
        let front_right = roof
            .keypoint(KeypointId::CrownTopFrontRight)
            .unwrap()
            .position;
        let back_left = roof
            .keypoint(KeypointId::CrownTopBackLeft)
            .unwrap()
            .position;

        assert_eq!(front_right[0] - front_left[0], parameters.crown_top_width);
        assert_eq!(back_left[2] - front_left[2], parameters.crown_top_depth);
        assert_eq!(front_left[1], parameters.lower_rise + parameters.upper_rise);

        let crown_top = roof.face(FaceId::CrownTop).unwrap();
        assert_eq!(crown_top.class, FaceClass::UpperCrown);
    }

    #[test]
    fn geometry_round_trips_through_json() {
        let roof = geometry();
        let json = serde_json::to_string(&roof).unwrap();
        let decoded: RoofGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, roof);
        assert_eq!(decoded.schema_version, GEOMETRY_SCHEMA_VERSION);
    }

    #[test]
    fn invalid_parameters_do_not_generate_a_mesh() {
        let parameters = RoofParameters {
            crown_top_width: f32::INFINITY,
            ..RoofParameters::default()
        };
        assert!(matches!(
            generate_roof(&parameters),
            Err(ParameterError::NonFinite {
                field: crate::ParameterField::CrownTopWidth,
                ..
            })
        ));
    }
}
