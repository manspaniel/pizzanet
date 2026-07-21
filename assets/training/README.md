# Training-data curation records

`open-images-review-ledger.json` is the digest-bound visual review of all 1,161
deterministically selected Open Images building candidates. It records the 288
rejections needed to recreate the 873 ordinary-building negatives without
repeating manual review.

After `roof-data import-open-images`, apply and verify it with:

```bash
cargo run --release -p roof-data -- apply-open-images-review \
  --ledger assets/training/open-images-review-ledger.json
cargo run --release -p roof-data -- validate-open-images-review
```

Image pixels, generated review sheets, and dataset manifests remain under the
gitignored `datasets/` directory. See `HANDOFF.md` and
`notes/TRAINING_AND_INFERENCE.md` for transfer and regeneration details.
