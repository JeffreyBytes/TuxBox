use super::*;

#[test]
fn counter_follows_creation_and_release() {
    let counter = Counter::new();
    counter.add(3 << 20);
    counter.add(1 << 20);
    assert_eq!(counter.read(), [2, 4 << 20, 2, 0]);
    counter.sub(3 << 20);
    assert_eq!(counter.read(), [1, 1 << 20, 2, 1]);
    counter.sub(1 << 20);
    assert_eq!(counter.read(), [0, 0, 2, 2]);
}

#[test]
fn mebibytes_keep_one_truncated_decimal() {
    assert_eq!(mib(0), "0.0M");
    assert_eq!(mib((1 << 20) + (1 << 19)), "1.5M");
    assert_eq!(mib((12 << 30) - 1), "12287.9M");
}
