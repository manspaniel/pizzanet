# Synthetic Rendering Assets

This directory contains a compact, repository-owned set of public-domain
assets for native synthetic-data rendering. The files are original 1K Poly
Haven downloads; they have not been resized, recompressed, or colour-corrected.

## Contents

| Asset | Author(s) | Intended variation | Files |
| --- | --- | --- | --- |
| Corrugated Iron 02 | Jenelle van Heerden; Sergej Majboroda | Galvanized or painted metal roof | diffuse, OpenGL normal, ARM |
| Roof 07 | Rob Tuytel | Weathered clay-tile roof | diffuse, OpenGL normal, ARM |
| Brick Wall 001 | Dimitrios Savva; Rob Tuytel | Red-brick building facade | diffuse, OpenGL normal, ARM |
| Clean Asphalt | Dimitrios Savva | Parking area and road | diffuse, OpenGL normal, ARM |
| Kloofendal 43d Clear (Pure Sky) | Greg Zaal | Clear, hard-sun daylight | HDR environment |
| Snow Field (Pure Sky) | Jarod Guest; Sergej Majboroda | Soft overcast daylight | HDR environment |
| Urban Street 04 | Andreas Mischok | Daytime urban street | HDR environment |
| Twilight Sunset | Dimitrios Savva; Jarod Guest | Urban/suburban twilight | HDR environment |
| Dikhololo Sunset | Greg Zaal | Roadside/remote twilight | HDR environment |
| Modern Buildings Night | Greg Zaal | Night-time urban environment | HDR environment |
| Goegap | Greg Zaal | Remote desert daylight | HDR environment |
| Kloppenheim 02 | Greg Zaal | Remote moonlit night | HDR environment |

The 20 files total 18,328,942 bytes (approximately 18.3 MB). `manifest.json`
records every exact download URL, author and role, source page, byte length,
SHA-256 digest, colour space, channel layout, and useful scene categories.

## License and provenance

Every included asset is published by [Poly Haven](https://polyhaven.com/) under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
Poly Haven's [asset-license statement](https://polyhaven.com/license) explicitly
permits use, modification, and redistribution without attribution. Attribution
is nevertheless retained in `manifest.json` so generated datasets remain fully
traceable. The manifest was captured on 2026-07-20 from the individual official
asset and download pages.

Do not add previews, user renders, logos, or other Poly Haven site content here:
the CC0 grant applies to the downloadable assets, not the surrounding website.

## WGPU integration

Material images are seamless 1024×1024 JPEGs. Decode them on the CPU and expand
RGB to RGBA before upload:

- diffuse maps are colour data and should use `Rgba8UnormSrgb`;
- normal and ARM maps are linear data and should use `Rgba8Unorm`;
- normals use the OpenGL convention (positive Y/green);
- ARM channels are red = ambient occlusion, green = roughness, and blue =
  metalness.

Use repeat addressing and generate mip levels. Colour randomisation may tint the
diffuse roof maps while leaving their normals and ARM values unchanged.

Environment files are 1024×512 Radiance HDR equirectangular panoramas in linear
RGB. Decode to floating-point pixels, add alpha, and upload as `Rgba16Float` or
`Rgba32Float`. Sample with repeat on longitude and clamp on latitude. They may be
used both as visible backgrounds and as inputs to an environment-lighting
prefilter. Day, twilight, and night each use phase-matched panoramas; twilight
selects urban/suburban or roadside/remote sunset environments by scene domain.

## Integrity checks

The Rust catalog loader requires the stable pack ID, checks that all declared
file lengths sum to `total_asset_bytes`, and verifies each on-disk length before
its SHA-256 digest and decoded dimensions.

ImageMagick can verify that every image decodes:

```bash
magick identify assets/synthetic/materials/*/* assets/synthetic/environments/*
```

Run the manifest hashes from the repository root:

```bash
jq -r '.assets[].files[] | [.sha256, ("assets/synthetic/" + .path)] | @tsv' \
  assets/synthetic/manifest.json | while IFS=$'\t' read -r digest path; do
    printf '%s  %s\n' "$digest" "$path"
  done | shasum -a 256 -c -
```
