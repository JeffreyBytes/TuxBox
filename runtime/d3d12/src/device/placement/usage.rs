//! `TUXBOX_TRACE_GPU_MEMORY=1`: which native copies of guest textures the
//! title still uses.
//!
//! Local diagnostic. Each placed image carries a record of atomic timestamps
//! only when the switch is set; without it every hook is a `None` check. The
//! records are aggregated when the periodic report is printed.
//!
//! * used: the image was in the texture set of a queue submission
//!   ([`super::submit`]): a copy source or destination, a cleared or drawn
//!   render target, or the target of a descriptor inside a descriptor table
//!   range that a draw or dispatch of the batch had bound.
//! * viewed: an SRV, UAV or RTV was created over the image.
//! * checked: the provider read the image's guest bytes and compared them with
//!   its snapshot. uploaded: that comparison differed and the image was
//!   refreshed (the first upload included).
use super::{FLUSH_DEVICES, Texture, lock};
use crate::diagnostics::gpu_memory::{self, Slot, mib};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, TryLockError};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Usage {
    /// Approximate bytes of the native image.
    bytes: u64,
    /// Creation order, to tell which of two overlapping images is newer.
    sequence: u64,
    // Milliseconds since the trace started plus one. Zero is never.
    created: u64,
    used: AtomicU64,
    used_by_command: AtomicU64,
    viewed: AtomicU64,
    checked: AtomicU64,
    uploaded: AtomicU64,
    uploads: AtomicU32,
    /// Title resource objects that currently hold the image.
    title: AtomicU32,
    titled: AtomicBool,
    /// The last check could not read some mip from guest memory.
    unreadable: AtomicBool,
}

fn stamp() -> u64 {
    gpu_memory::now_ms() + 1
}

impl Usage {
    pub(super) fn new(native: &ID3D12Resource) -> Option<Arc<Self>> {
        let bytes = gpu_memory::guest_texture_copy(native)?;
        Some(Arc::new(Self {
            bytes,
            sequence: SEQUENCE.fetch_add(1, Relaxed),
            created: stamp(),
            used: AtomicU64::new(0),
            used_by_command: AtomicU64::new(0),
            viewed: AtomicU64::new(0),
            checked: AtomicU64::new(0),
            uploaded: AtomicU64::new(0),
            uploads: AtomicU32::new(0),
            title: AtomicU32::new(0),
            titled: AtomicBool::new(false),
            unreadable: AtomicBool::new(false),
        }))
    }
}

/// One title resource object holding a placed image.
pub(crate) struct TitleRef(Arc<Usage>);

impl Drop for TitleRef {
    fn drop(&mut self) {
        self.0.title.fetch_sub(1, Relaxed);
    }
}

impl Texture {
    pub(crate) fn title_ref(&self) -> Option<TitleRef> {
        let usage = self.usage.as_ref()?;
        usage.title.fetch_add(1, Relaxed);
        usage.titled.store(true, Relaxed);
        Some(TitleRef(usage.clone()))
    }

    pub(crate) fn viewed(&self) {
        if let Some(usage) = &self.usage {
            usage.viewed.store(stamp(), Relaxed);
        }
    }

    pub(super) fn checked(&self, readable: bool) {
        if let Some(usage) = &self.usage {
            usage.checked.store(stamp(), Relaxed);
            usage.unreadable.store(!readable, Relaxed);
        }
    }

    /// One changed mip was uploaded. Mips of one refresh share a timestamp.
    pub(super) fn uploaded(&self) {
        if let Some(usage) = &self.usage {
            let now = stamp();
            if usage.uploaded.swap(now, Relaxed) != now {
                usage.uploads.fetch_add(1, Relaxed);
            }
        }
    }
}

/// The placed images of one queue submission. `by_command` marks the ones a
/// recorded command named itself, before descriptor tables are resolved.
pub(super) fn submitted(reads: &[Arc<Texture>], writes: &[Arc<Texture>], by_command: bool) {
    if !gpu_memory::enabled() {
        return;
    }
    let now = stamp();
    for usage in reads.iter().chain(writes).filter_map(|texture| texture.usage.as_ref()) {
        if by_command {
            usage.used_by_command.store(now, Relaxed);
        }
        usage.used.store(now, Relaxed);
    }
}

/// Images and bytes.
#[derive(Clone, Copy, Default)]
struct Tally(u64, u64);

impl Tally {
    fn add(&mut self, bytes: u64) {
        self.0 += 1;
        self.1 += bytes;
    }
}

impl core::fmt::Display for Tally {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}/{}", self.0, mib(self.1))
    }
}

/// Never, under 5 s, under 30 s, under 120 s, older.
pub(super) fn age_bucket(now: u64, stamp: u64) -> usize {
    match now.saturating_sub(stamp) {
        _ if stamp == 0 => 0,
        0..5_000 => 1,
        5_000..30_000 => 2,
        30_000..120_000 => 3,
        _ => 4,
    }
}

/// Under 64 KiB, under 1 MiB, under 16 MiB, larger.
pub(super) fn size_bucket(bytes: u64) -> usize {
    match bytes {
        0..0x1_0000 => 0,
        0x1_0000..0x10_0000 => 1,
        0x10_0000..0x100_0000 => 2,
        _ => 3,
    }
}

pub(super) const OVERLAPPED: u8 = 1;
pub(super) const SUPERSEDED: u8 = 2;
// A bound on the neighbors examined per image, so one huge range over many
// small ones cannot make the report quadratic.
const NEIGHBORS: usize = 256;

/// For guest ranges `(start, end, sequence)` sorted by start: whether each
/// overlaps another range, and whether one that overlaps it is newer.
pub(super) fn overlap_flags(ranges: &[(u64, u64, u64)]) -> Vec<u8> {
    let mut flags = vec![0u8; ranges.len()];
    for (i, a) in ranges.iter().enumerate() {
        for (j, b) in ranges.iter().enumerate().skip(i + 1).take(NEIGHBORS) {
            if b.0 >= a.1 {
                break;
            }
            flags[i] |= OVERLAPPED | if b.2 > a.2 { SUPERSEDED } else { 0 };
            flags[j] |= OVERLAPPED | if a.2 > b.2 { SUPERSEDED } else { 0 };
        }
    }
    flags
}

fn ages(tallies: &[Tally; 5]) -> String {
    format!(
        "never={} <5s={} <30s={} <120s={} >120s={}",
        tallies[0], tallies[1], tallies[2], tallies[3], tallies[4],
    )
}

/// Lines appended to the periodic GPU memory report. Each value is
/// `images/MiB of native image` unless labelled otherwise.
pub(crate) fn report(log: &dyn Fn(String)) {
    let devices: Vec<_> = lock(&FLUSH_DEVICES).iter().filter_map(std::sync::Weak::upgrade).collect();
    // Upgraded under the table lock and dropped after it: a last reference
    // dropped here runs `Texture::drop`, which takes that lock.
    let mut textures: Vec<Arc<Texture>> = Vec::new();
    for device in &devices {
        textures.extend(lock(&device.images.0).values().flatten().filter_map(|alias| alias.texture.upgrade()));
    }
    textures.retain(|texture| texture.usage.is_some());
    textures.sort_unstable_by_key(|texture| (texture.address, texture.usage.as_ref().map(|u| u.sequence)));
    let ranges: Vec<_> = textures
        .iter()
        .map(|t| (t.address, t.address.saturating_add(t.size), t.usage.as_ref().map_or(0, |u| u.sequence)))
        .collect();
    let overlaps = overlap_flags(&ranges);

    let now = stamp();
    let mut live = Tally::default();
    let (mut use_age, mut upload_age) = ([Tally::default(); 5], [Tally::default(); 5]);
    let (mut never_old, mut by_command, mut viewed, mut checked, mut reuploaded) =
        (Tally::default(), Tally::default(), Tally::default(), Tally::default(), Tally::default());
    let (mut released, mut untitled, mut unreadable, mut unmapped, mut superseded, mut overlapped) =
        (Tally::default(), Tally::default(), Tally::default(), Tally::default(), Tally::default(), Tally::default());
    let mut sizes = [Tally::default(); 4];
    // Snapshot tallies count images and bytes of host RAM, not image bytes.
    let (mut snapshots, mut idle_snapshots, mut orphan_snapshots) = (Tally::default(), Tally::default(), Tally::default());
    let (mut locked, mut addresses, mut run, mut most, mut last) = (0u64, 0u64, 0u64, 0u64, None);

    for (texture, flags) in textures.iter().zip(&overlaps) {
        let Some(usage) = &texture.usage else { continue };
        let bytes = usage.bytes;
        live.add(bytes);
        let used = usage.used.load(Relaxed);
        let uploaded = usage.uploaded.load(Relaxed);
        use_age[age_bucket(now, used)].add(bytes);
        upload_age[age_bucket(now, uploaded)].add(bytes);
        // Idle: no use for 30 s (or none since a creation 30 s ago).
        let idle = age_bucket(now, used.max(usage.created)) >= 3;
        if used == 0 && idle {
            never_old.add(bytes);
        }
        if age_bucket(now, usage.used_by_command.load(Relaxed)) == 1 {
            by_command.add(bytes);
        }
        if age_bucket(now, usage.viewed.load(Relaxed)) == 1 {
            viewed.add(bytes);
        }
        if age_bucket(now, usage.checked.load(Relaxed)) == 1 {
            checked.add(bytes);
        }
        if usage.uploads.load(Relaxed) > 1 {
            reuploaded.add(bytes);
        }
        let orphan = usage.title.load(Relaxed) == 0;
        if orphan {
            if usage.titled.load(Relaxed) { released.add(bytes) } else { untitled.add(bytes) }
        }
        if usage.unreadable.load(Relaxed) {
            unreadable.add(bytes);
        }
        if !super::writeback::query_region(texture.address as usize)
            .is_ok_and(|(_, info)| super::writeback::mapped_region(&info))
        {
            unmapped.add(bytes);
        }
        if flags & SUPERSEDED != 0 {
            superseded.add(bytes);
        }
        if flags & OVERLAPPED != 0 {
            overlapped.add(bytes);
        }
        sizes[size_bucket(bytes)].add(bytes);
        if last == Some(texture.address) {
            run += 1;
        } else {
            (addresses, run, last) = (addresses + 1, 1, Some(texture.address));
        }
        most = most.max(run);
        // Never wait for a queue or publication thread that holds the contents.
        let held = match texture.contents.try_lock() {
            Ok(contents) => Some(contents.previous.iter().map(|mip| mip.capacity() as u64).sum::<u64>()),
            Err(TryLockError::Poisoned(error)) => {
                Some(error.into_inner().previous.iter().map(|mip| mip.capacity() as u64).sum())
            }
            Err(TryLockError::WouldBlock) => None,
        };
        match held {
            None => locked += 1,
            Some(0) => {}
            Some(held) => {
                snapshots.add(held);
                if idle && age_bucket(now, uploaded) >= 3 {
                    idle_snapshots.add(held);
                }
                if orphan {
                    orphan_snapshots.add(held);
                }
            }
        }
    }

    let [native, native_bytes] = gpu_memory::guest_texture_copies();
    let native_only = Tally(native.saturating_sub(live.0), native_bytes.saturating_sub(live.1));
    log(format!(
        "guest tex use age: {} | never&old>30s={never_old} cmd<5s={by_command} view<5s={viewed}",
        ages(&use_age),
    ));
    log(format!(
        "guest tex upload age: {} | checked<5s={checked} reuploaded={reuploaded}",
        ages(&upload_age),
    ));
    log(format!(
        "guest tex owners: live={live} title_refs=0 released={released} never_titled={untitled} unreadable={unreadable} va_gone={unmapped} superseded={superseded} overlapped={overlapped} native_only={native_only}",
    ));
    log(format!(
        "guest tex size: <64K={} <1M={} <16M={} >=16M={} | addrs={addresses} images={} max_per_addr={most}",
        sizes[0], sizes[1], sizes[2], sizes[3], live.0,
    ));
    log(format!(
        "guest tex snapshots (images/host MiB): alloc={snapshots} bound={} idle>30s={idle_snapshots} title_refs=0 {orphan_snapshots} locked={locked}",
        mib(gpu_memory::host_ram_bytes(Slot::SnapshotRam)),
    ));
}

#[cfg(test)]
#[path = "../../tests/device/placement_usage.rs"]
mod tests;
