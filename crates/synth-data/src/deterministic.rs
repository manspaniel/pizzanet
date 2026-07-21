//! Stable hashing and seed derivation used by split assignment and sampling.

/// Hashes length-delimited byte slices with a fixed, repository-owned algorithm.
///
/// This is not a cryptographic hash. Its purpose is to avoid Rust's deliberately
/// unstable `HashMap` hasher in durable dataset decisions.
pub(crate) fn stable_hash64(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for part in parts {
        for byte in (part.len() as u64).to_le_bytes().iter().chain(part.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    // MurmurHash3's final avalanche prevents structured low bits from biasing
    // the basis-point buckets used for dataset splits.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

pub(crate) fn derive_seed(root_seed: u64, domain: &str) -> u64 {
    stable_hash64(&[
        b"synth-data-seed-v1",
        &root_seed.to_le_bytes(),
        domain.as_bytes(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_length_delimited() {
        assert_ne!(stable_hash64(&[b"ab", b"c"]), stable_hash64(&[b"a", b"bc"]));
    }

    #[test]
    fn domain_seeds_are_independent() {
        assert_ne!(derive_seed(42, "roof"), derive_seed(42, "camera"));
        assert_ne!(derive_seed(42, "roof"), derive_seed(43, "roof"));
    }
}
