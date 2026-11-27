use super::*;

#[test]
fn ages_fall_in_the_reported_buckets() {
    let now = 200_000;
    assert_eq!(age_bucket(now, 0), 0);
    assert_eq!(age_bucket(now, now), 1);
    assert_eq!(age_bucket(now, now - 4_999), 1);
    assert_eq!(age_bucket(now, now - 5_000), 2);
    assert_eq!(age_bucket(now, now - 29_999), 2);
    assert_eq!(age_bucket(now, now - 30_000), 3);
    assert_eq!(age_bucket(now, now - 120_000), 4);
    // A stamp taken after `now` on another thread is fresh, not old.
    assert_eq!(age_bucket(now, now + 3), 1);
}

#[test]
fn sizes_fall_in_the_reported_buckets() {
    assert_eq!(size_bucket(0), 0);
    assert_eq!(size_bucket((64 << 10) - 1), 0);
    assert_eq!(size_bucket(64 << 10), 1);
    assert_eq!(size_bucket(1 << 20), 2);
    assert_eq!(size_bucket((16 << 20) - 1), 2);
    assert_eq!(size_bucket(16 << 20), 3);
}

#[test]
fn overlap_marks_both_ranges_and_the_older_one_as_superseded() {
    // (start, end, creation sequence), sorted by start.
    let ranges = [
        (0x1000, 0x2000, 5), // same address as the next, created later
        (0x1000, 0x3000, 1), // superseded by the first and by the third
        (0x2800, 0x4000, 7), // newest of its overlaps
        (0x4000, 0x5000, 0), // adjacent, not overlapping
    ];
    assert_eq!(
        overlap_flags(&ranges),
        [OVERLAPPED, OVERLAPPED | SUPERSEDED, OVERLAPPED, 0],
    );
    assert!(overlap_flags(&[]).is_empty());
}
