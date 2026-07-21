# Reference Image Calibration

The local `samples/` photographs are an architectural calibration set, not one
canonical building. The generator must preserve the shared two-stage form while
sampling the large differences in footprint, pitch break, crown prominence,
finish, façade, signage, and later conversion work visible across the set.

The image that first exposed the missing tier (`b8bn57ak1kp71.jpg`) is only one
member of the set. All 18 references were reviewed together.

## Morphology groups

The images support three correlated parameter families. These are sampling
envelopes, not hard historical labels:

- `tall_early_crown`: compact or moderate footprints with a conspicuous steep
  upper crown. Strong examples include `531602ef6bb3f7b52ae2ea7a.webp`,
  `b8bn57ak1kp71.jpg`, `images-3.jpg`, `images-4.jpg`, `images-5.jpg`,
  `r0_205_775_640_w1200_h678_fmax.jpg`, and
  `why-was-pizza-hut-shaped-like-this-v0-idg5oe5x1c7d1.webp`.
- `balanced_classic`: the two stages have comparable visual weight and a
  moderate footprint aspect. Examples include
  `2b969810-2dc9-446f-a37e-617a73fe7b2e_1002x758.jpg`, `images-2.jpg`,
  `images-7.jpg`, and `pizzahutAP060206020442.webp`.
- `low_wide_late`: wider footprints, lower crowns, or heavily altered former
  restaurants where the silhouette remains recognisable. Examples include
  `47-612e3f4cc0fb7__700.jpg`, `5315f5be69bedddd37222340.webp`,
  `97034e94af7a1a3f721aae5c03f225e5.jpg`, `images-1.jpg`, `images-6.jpg`,
  `images.jpg`, and `pizza-hut-scaled-e1780056095642-1024x478.jpg`.

`crates/synth-data/src/config.rs` encodes these as weighted
`RoofMorphologyProfile` values. A profile is selected before dimensions, so
footprint aspect, overhang, shoulder inset, lower rise, upper rise, and crown
top proportions vary together instead of producing implausible independent
combinations. The selected morphology is persisted in every sequence record.

## Appearance and context

Roof colour is deliberately independent of morphology. The references include
red tile and metal, faded brown, green, blue, black, grey, and yellow repainting.
They also include active Pizza Hut branding, sign ghosts, unrelated tenants,
blank façades, boarded windows, extensions, and substantial remodelling.
Neither colour nor signage may become a shortcut for the recognition label.

The photographs span archival and modern cameras, close and distant framing,
front and oblique views, trees, utility lines, roads, car parks, neighbouring
buildings, and partial obstruction. Synthetic sampling combines those cues
with the independent day/twilight/night and city/urban/suburban/roadside/remote
regimes rather than copying a photographed background.

## Data boundary

These local images guide morphology and real-image evaluation. Their pixels are
not redistributed as part of the CC0 synthetic asset pack and are not silently
used as HDR backgrounds or textures. Any use as model training imagery must
carry its own provenance and physical-building split group. Generated-gallery
acceptance checks each morphology across varied materials, framing, lighting,
and occlusion while retaining exact structural labels from the shared mesh.
