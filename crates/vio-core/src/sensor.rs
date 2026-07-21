use crate::{DVec3, MonotonicDuration, MonotonicTimestamp};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use thiserror::Error;

/// A monotonically increasing motion-event identifier within one acquisition epoch.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SensorSequence(u64);

impl SensorSequence {
    /// Constructs a sequence identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The screen orientation reported with a raw event before its vectors were normalized.
///
/// This remains useful diagnostic metadata. It does not change the coordinate convention of
/// [`NormalizedMotionSample`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceScreenOrientation {
    /// Upright portrait orientation.
    #[default]
    PortraitPrimary,
    /// Upside-down portrait orientation.
    PortraitSecondary,
    /// Landscape with the device rotated counter-clockwise from portrait-primary.
    LandscapePrimary,
    /// Landscape with the device rotated clockwise from portrait-primary.
    LandscapeSecondary,
    /// The acquisition source could not report a stable orientation.
    Unknown,
}

/// One normalized DeviceMotion-style sample.
///
/// Vectors use SI units in a right-handed portrait-primary device body frame: +X points toward
/// the right edge, +Y toward the top edge, and +Z out of the screen. Angular velocity is in
/// radians/second. `specific_force_mps2` is proper accelerometer force, not gravity-removed linear
/// acceleration; a motionless level device therefore measures approximately +9.80665 on an
/// upward-facing body axis. The state propagation equation is
/// `world_acceleration = R * (specific_force - bias) + world_gravity`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedMotionSample {
    sequence: SensorSequence,
    event_timestamp: MonotonicTimestamp,
    receipt_timestamp: MonotonicTimestamp,
    reported_interval: Option<MonotonicDuration>,
    angular_velocity_rad_s: DVec3,
    specific_force_mps2: DVec3,
    linear_acceleration_mps2: Option<DVec3>,
    source_screen_orientation: SourceScreenOrientation,
}

impl NormalizedMotionSample {
    /// Constructs a sample after platform-specific unit, sign, and orientation normalization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: SensorSequence,
        event_timestamp: MonotonicTimestamp,
        receipt_timestamp: MonotonicTimestamp,
        reported_interval: Option<MonotonicDuration>,
        angular_velocity_rad_s: DVec3,
        specific_force_mps2: DVec3,
        linear_acceleration_mps2: Option<DVec3>,
        source_screen_orientation: SourceScreenOrientation,
    ) -> Result<Self, SampleError> {
        if !angular_velocity_rad_s.is_finite()
            || !specific_force_mps2.is_finite()
            || linear_acceleration_mps2.is_some_and(|value| !value.is_finite())
        {
            return Err(SampleError::NonFiniteVector);
        }

        Ok(Self {
            sequence,
            event_timestamp,
            receipt_timestamp,
            reported_interval,
            angular_velocity_rad_s,
            specific_force_mps2,
            linear_acceleration_mps2,
            source_screen_orientation,
        })
    }

    /// Returns the event sequence.
    #[must_use]
    pub const fn sequence(&self) -> SensorSequence {
        self.sequence
    }

    /// Returns the timestamp carried by the source event.
    #[must_use]
    pub const fn event_timestamp(&self) -> MonotonicTimestamp {
        self.event_timestamp
    }

    /// Returns the timestamp taken immediately when the event handler received the sample.
    #[must_use]
    pub const fn receipt_timestamp(&self) -> MonotonicTimestamp {
        self.receipt_timestamp
    }

    /// Returns the source-reported sampling interval when available.
    #[must_use]
    pub const fn reported_interval(&self) -> Option<MonotonicDuration> {
        self.reported_interval
    }

    /// Returns normalized gyroscope angular velocity in radians/second.
    #[must_use]
    pub const fn angular_velocity_rad_s(&self) -> DVec3 {
        self.angular_velocity_rad_s
    }

    /// Returns normalized proper accelerometer force in metres/second squared.
    #[must_use]
    pub const fn specific_force_mps2(&self) -> DVec3 {
        self.specific_force_mps2
    }

    /// Returns the platform's gravity-removed acceleration, retained only as a diagnostic signal.
    #[must_use]
    pub const fn linear_acceleration_mps2(&self) -> Option<DVec3> {
        self.linear_acceleration_mps2
    }

    /// Returns the screen orientation associated with the unnormalized source event.
    #[must_use]
    pub const fn source_screen_orientation(&self) -> SourceScreenOrientation {
        self.source_screen_orientation
    }

    /// Selects one of the preserved timestamps.
    #[must_use]
    pub const fn timestamp(&self, basis: SensorTimeBasis) -> MonotonicTimestamp {
        match basis {
            SensorTimeBasis::Event => self.event_timestamp,
            SensorTimeBasis::Receipt => self.receipt_timestamp,
        }
    }
}

/// Invalid normalized sample data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum SampleError {
    /// At least one vector component was NaN or infinite.
    #[error("motion vectors must contain only finite values")]
    NonFiniteVector,
}

/// Chooses which preserved clock value orders and associates sensor samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorTimeBasis {
    /// Prefer the timestamp carried by the source event.
    #[default]
    Event,
    /// Prefer the timestamp recorded by the receiving event handler.
    Receipt,
}

/// A camera interval using the exact half-open/half-closed convention `(start, end]`.
///
/// A sample at `start` belongs to the previous frame and may be retained as integration context.
/// A sample at `end` belongs to this interval. This prevents boundary samples being consumed twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameInterval {
    start_exclusive: MonotonicTimestamp,
    end_inclusive: MonotonicTimestamp,
}

impl FrameInterval {
    /// Constructs a strictly increasing camera interval.
    pub fn new(
        start_exclusive: MonotonicTimestamp,
        end_inclusive: MonotonicTimestamp,
    ) -> Result<Self, BufferError> {
        if end_inclusive <= start_exclusive {
            return Err(BufferError::InvalidInterval);
        }
        Ok(Self {
            start_exclusive,
            end_inclusive,
        })
    }

    /// Returns the excluded start timestamp.
    #[must_use]
    pub const fn start_exclusive(self) -> MonotonicTimestamp {
        self.start_exclusive
    }

    /// Returns the included end timestamp.
    #[must_use]
    pub const fn end_inclusive(self) -> MonotonicTimestamp {
        self.end_inclusive
    }

    /// Returns the interval duration.
    #[must_use]
    pub fn duration(self) -> MonotonicDuration {
        self.end_inclusive
            .checked_duration_since(self.start_exclusive)
            .expect("FrameInterval construction guarantees ordered bounds")
    }

    /// Reports whether a timestamp is in `(start, end]`.
    #[must_use]
    pub fn contains(self, timestamp: MonotonicTimestamp) -> bool {
        timestamp > self.start_exclusive && timestamp <= self.end_inclusive
    }
}

/// Loss counters associated with one emitted sensor batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchDiagnostics {
    /// Samples discarded by bounded capacity since the previous batch.
    pub capacity_drops: u64,
    /// Samples rejected behind the consumed watermark since the previous batch.
    pub late_rejections: u64,
    /// Buffered samples at or before the requested start that were not previously consumed.
    pub stale_samples: u64,
}

/// An immutable ordered sensor batch associated with one camera interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorBatch {
    time_basis: SensorTimeBasis,
    interval: FrameInterval,
    leading_sample: Option<NormalizedMotionSample>,
    samples: Vec<NormalizedMotionSample>,
    trailing_sample: Option<NormalizedMotionSample>,
    diagnostics: BatchDiagnostics,
}

impl SensorBatch {
    /// Constructs a validated batch, primarily for deterministic replay and tests.
    pub fn new(
        time_basis: SensorTimeBasis,
        interval: FrameInterval,
        leading_sample: Option<NormalizedMotionSample>,
        samples: Vec<NormalizedMotionSample>,
        trailing_sample: Option<NormalizedMotionSample>,
        diagnostics: BatchDiagnostics,
    ) -> Result<Self, BufferError> {
        if leading_sample
            .as_ref()
            .is_some_and(|sample| sample.timestamp(time_basis) > interval.start_exclusive)
            || trailing_sample
                .as_ref()
                .is_some_and(|sample| sample.timestamp(time_basis) <= interval.end_inclusive)
            || samples
                .iter()
                .any(|sample| !interval.contains(sample.timestamp(time_basis)))
        {
            return Err(BufferError::SampleOutsideInterval);
        }

        if samples
            .windows(2)
            .any(|pair| compare_samples(&pair[0], &pair[1], time_basis).is_gt())
        {
            return Err(BufferError::SamplesOutOfOrder);
        }

        Ok(Self {
            time_basis,
            interval,
            leading_sample,
            samples,
            trailing_sample,
            diagnostics,
        })
    }

    /// Returns the clock basis used for ordering and interval membership.
    #[must_use]
    pub const fn time_basis(&self) -> SensorTimeBasis {
        self.time_basis
    }

    /// Returns the associated camera interval.
    #[must_use]
    pub const fn interval(&self) -> FrameInterval {
        self.interval
    }

    /// Returns the newest sample at or before the interval start, if retained.
    #[must_use]
    pub const fn leading_sample(&self) -> Option<&NormalizedMotionSample> {
        self.leading_sample.as_ref()
    }

    /// Returns samples in `(start, end]`, ordered deterministically.
    #[must_use]
    pub fn samples(&self) -> &[NormalizedMotionSample] {
        &self.samples
    }

    /// Returns the earliest buffered sample after the interval, if already available.
    #[must_use]
    pub const fn trailing_sample(&self) -> Option<&NormalizedMotionSample> {
        self.trailing_sample.as_ref()
    }

    /// Returns loss counters captured for this interval.
    #[must_use]
    pub const fn diagnostics(&self) -> BatchDiagnostics {
        self.diagnostics
    }
}

/// The result of inserting one sensor sample into a bounded buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushOutcome {
    /// The sample was inserted without loss.
    Inserted,
    /// The sample was inserted and an older buffered sample was discarded.
    InsertedAndDroppedOldest {
        /// Sequence identifier of the discarded sample.
        dropped: SensorSequence,
    },
    /// The incoming sample was older than every retained sample and was discarded by capacity.
    RejectedByCapacity,
    /// The sample timestamp was at or behind the already emitted watermark.
    RejectedLate {
        /// The newest interval endpoint already emitted.
        watermark: MonotonicTimestamp,
    },
}

/// Cumulative bounded-buffer counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferStats {
    /// Samples successfully retained at insertion time.
    pub inserted: u64,
    /// Samples removed because the configured capacity was exceeded.
    pub capacity_drops: u64,
    /// Samples rejected because their interval had already been emitted.
    pub late_rejections: u64,
    /// Camera batches emitted.
    pub emitted_batches: u64,
}

/// A bounded, reorder-tolerant sensor buffer with deterministic frame batching.
#[derive(Clone, Debug)]
pub struct SensorBuffer {
    capacity: usize,
    time_basis: SensorTimeBasis,
    samples: Vec<NormalizedMotionSample>,
    carry: Option<NormalizedMotionSample>,
    drained_through: Option<MonotonicTimestamp>,
    stats: BufferStats,
    capacity_drops_since_batch: u64,
    late_rejections_since_batch: u64,
}

impl SensorBuffer {
    /// Constructs a buffer. Capacity excludes the one retained leading context sample.
    pub fn new(capacity: usize, time_basis: SensorTimeBasis) -> Result<Self, BufferError> {
        if capacity == 0 {
            return Err(BufferError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            time_basis,
            samples: Vec::with_capacity(capacity),
            carry: None,
            drained_through: None,
            stats: BufferStats::default(),
            capacity_drops_since_batch: 0,
            late_rejections_since_batch: 0,
        })
    }

    /// Inserts a sample in timestamp/sequence order, tolerating arrival reordering before drain.
    pub fn push(&mut self, sample: NormalizedMotionSample) -> PushOutcome {
        let timestamp = sample.timestamp(self.time_basis);
        if let Some(watermark) = self.drained_through
            && timestamp <= watermark
        {
            self.stats.late_rejections += 1;
            self.late_rejections_since_batch += 1;
            return PushOutcome::RejectedLate { watermark };
        }

        let insertion_index = self.samples.partition_point(|existing| {
            !compare_samples(existing, &sample, self.time_basis).is_gt()
        });
        self.samples.insert(insertion_index, sample);

        if self.samples.len() <= self.capacity {
            self.stats.inserted += 1;
            return PushOutcome::Inserted;
        }

        let dropped = self.samples.remove(0);
        self.stats.capacity_drops += 1;
        self.capacity_drops_since_batch += 1;
        if insertion_index == 0 {
            PushOutcome::RejectedByCapacity
        } else {
            self.stats.inserted += 1;
            PushOutcome::InsertedAndDroppedOldest {
                dropped: dropped.sequence(),
            }
        }
    }

    /// Emits all samples in `(start, end]` while retaining bracketing context when available.
    pub fn drain_interval(&mut self, interval: FrameInterval) -> Result<SensorBatch, BufferError> {
        if self
            .drained_through
            .is_some_and(|watermark| interval.start_exclusive < watermark)
        {
            return Err(BufferError::IntervalBeforeWatermark);
        }

        let consumed_count = self
            .samples
            .partition_point(|sample| sample.timestamp(self.time_basis) <= interval.end_inclusive);
        let consumed: Vec<_> = self.samples.drain(..consumed_count).collect();

        let mut leading = self
            .carry
            .take()
            .filter(|sample| sample.timestamp(self.time_basis) <= interval.start_exclusive);
        let mut samples = Vec::new();
        let mut stale_samples = 0_u64;

        for sample in consumed {
            if sample.timestamp(self.time_basis) <= interval.start_exclusive {
                leading = Some(sample);
                stale_samples += 1;
            } else {
                samples.push(sample);
            }
        }

        self.carry = samples.last().cloned().or_else(|| leading.clone());
        let trailing = self.samples.first().cloned();
        let diagnostics = BatchDiagnostics {
            capacity_drops: self.capacity_drops_since_batch,
            late_rejections: self.late_rejections_since_batch,
            stale_samples,
        };
        self.capacity_drops_since_batch = 0;
        self.late_rejections_since_batch = 0;
        self.drained_through = Some(interval.end_inclusive);
        self.stats.emitted_batches += 1;

        SensorBatch::new(
            self.time_basis,
            interval,
            leading,
            samples,
            trailing,
            diagnostics,
        )
    }

    /// Returns the selected ordering clock.
    #[must_use]
    pub const fn time_basis(&self) -> SensorTimeBasis {
        self.time_basis
    }

    /// Returns the number of not-yet-emitted samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Reports whether there are no not-yet-emitted samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns cumulative loss and throughput counters.
    #[must_use]
    pub const fn stats(&self) -> BufferStats {
        self.stats
    }

    /// Returns the most recent emitted endpoint.
    #[must_use]
    pub const fn drained_through(&self) -> Option<MonotonicTimestamp> {
        self.drained_through
    }
}

/// Invalid buffer configuration or batching request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BufferError {
    /// A bounded buffer must retain at least one sample.
    #[error("sensor buffer capacity must be greater than zero")]
    ZeroCapacity,
    /// A frame interval did not increase strictly.
    #[error("frame interval end must be later than its start")]
    InvalidInterval,
    /// A requested interval overlaps already emitted time.
    #[error("frame interval starts before the emitted sensor watermark")]
    IntervalBeforeWatermark,
    /// A replay batch contained a sample outside its declared interval/context position.
    #[error("sensor sample is outside the declared frame interval")]
    SampleOutsideInterval,
    /// A replay batch was not ordered by timestamp and sequence.
    #[error("sensor samples must be in deterministic timestamp order")]
    SamplesOutOfOrder,
}

fn compare_samples(
    left: &NormalizedMotionSample,
    right: &NormalizedMotionSample,
    basis: SensorTimeBasis,
) -> Ordering {
    left.timestamp(basis)
        .cmp(&right.timestamp(basis))
        .then_with(|| match basis {
            SensorTimeBasis::Event => left.receipt_timestamp.cmp(&right.receipt_timestamp),
            SensorTimeBasis::Receipt => left.event_timestamp.cmp(&right.event_timestamp),
        })
        .then_with(|| left.sequence.cmp(&right.sequence))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timestamp(milliseconds: u64) -> MonotonicTimestamp {
        MonotonicTimestamp::from_nanos(milliseconds * 1_000_000)
    }

    fn sample(sequence: u64, event_ms: u64, receipt_ms: u64) -> NormalizedMotionSample {
        NormalizedMotionSample::new(
            SensorSequence::new(sequence),
            timestamp(event_ms),
            timestamp(receipt_ms),
            Some(MonotonicDuration::from_nanos(10_000_000)),
            DVec3::new(sequence as f64, 0.0, 0.0),
            DVec3::Z * 9.806_65,
            Some(DVec3::ZERO),
            SourceScreenOrientation::PortraitPrimary,
        )
        .unwrap()
    }

    #[test]
    fn insertion_reorders_by_selected_timestamp_then_sequence() {
        let mut buffer = SensorBuffer::new(8, SensorTimeBasis::Event).unwrap();
        buffer.push(sample(3, 30, 31));
        buffer.push(sample(1, 10, 13));
        buffer.push(sample(2, 20, 22));

        let batch = buffer
            .drain_interval(FrameInterval::new(timestamp(0), timestamp(30)).unwrap())
            .unwrap();
        let sequences: Vec<_> = batch
            .samples()
            .iter()
            .map(|value| value.sequence().get())
            .collect();
        assert_eq!(sequences, [1, 2, 3]);
    }

    #[test]
    fn receipt_basis_can_differ_from_event_order() {
        let mut buffer = SensorBuffer::new(8, SensorTimeBasis::Receipt).unwrap();
        buffer.push(sample(1, 10, 25));
        buffer.push(sample(2, 20, 21));

        let batch = buffer
            .drain_interval(FrameInterval::new(timestamp(0), timestamp(30)).unwrap())
            .unwrap();
        assert_eq!(batch.samples()[0].sequence(), SensorSequence::new(2));
        assert_eq!(batch.samples()[1].sequence(), SensorSequence::new(1));
    }

    #[test]
    fn interval_boundaries_are_consumed_exactly_once() {
        let mut buffer = SensorBuffer::new(8, SensorTimeBasis::Event).unwrap();
        buffer.push(sample(1, 10, 10));
        buffer.push(sample(2, 20, 20));
        buffer.push(sample(3, 30, 30));

        let first = buffer
            .drain_interval(FrameInterval::new(timestamp(10), timestamp(20)).unwrap())
            .unwrap();
        assert_eq!(
            first.leading_sample().unwrap().sequence(),
            SensorSequence::new(1)
        );
        assert_eq!(first.samples().len(), 1);
        assert_eq!(first.samples()[0].sequence(), SensorSequence::new(2));

        let second = buffer
            .drain_interval(FrameInterval::new(timestamp(20), timestamp(30)).unwrap())
            .unwrap();
        assert_eq!(
            second.leading_sample().unwrap().sequence(),
            SensorSequence::new(2)
        );
        assert_eq!(second.samples().len(), 1);
        assert_eq!(second.samples()[0].sequence(), SensorSequence::new(3));
    }

    #[test]
    fn capacity_and_late_loss_are_explicit() {
        let mut buffer = SensorBuffer::new(2, SensorTimeBasis::Event).unwrap();
        buffer.push(sample(1, 10, 10));
        buffer.push(sample(2, 20, 20));
        assert_eq!(
            buffer.push(sample(3, 30, 30)),
            PushOutcome::InsertedAndDroppedOldest {
                dropped: SensorSequence::new(1)
            }
        );

        let interval = FrameInterval::new(timestamp(0), timestamp(30)).unwrap();
        let batch = buffer.drain_interval(interval).unwrap();
        assert_eq!(batch.diagnostics().capacity_drops, 1);
        assert_eq!(
            buffer.push(sample(4, 25, 40)),
            PushOutcome::RejectedLate {
                watermark: timestamp(30)
            }
        );
    }

    #[test]
    fn motion_batch_round_trips_through_serde() {
        let interval = FrameInterval::new(timestamp(10), timestamp(20)).unwrap();
        let batch = SensorBatch::new(
            SensorTimeBasis::Event,
            interval,
            Some(sample(1, 10, 11)),
            vec![sample(2, 15, 17), sample(3, 20, 22)],
            Some(sample(4, 25, 27)),
            BatchDiagnostics::default(),
        )
        .unwrap();

        let encoded = serde_json::to_string(&batch).unwrap();
        let decoded: SensorBatch = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, batch);
        assert!(encoded.contains("event_timestamp"));
        assert!(encoded.contains("receipt_timestamp"));
    }
}
