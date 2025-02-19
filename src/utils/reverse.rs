use std::cmp::Ordering;

#[derive(PartialEq, Eq)]
pub struct Reverse<N>(pub N);

impl<N: Ord> Ord for Reverse<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl<N: Ord> PartialOrd for Reverse<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_ord() {
        let a = Reverse(5);
        let b = Reverse(10);

        assert!(b < a); // Теперь тестируем, что Reverse(10) должно быть меньше Reverse(5)
        assert!(a > b); // И наоборот
    }

    #[test]
    fn test_reverse_eq() {
        let a = Reverse(5);
        let b = Reverse(5);

        assert!(a == b); // Должны быть равны
    }

    #[test]
    fn test_reverse_partial_ord() {
        let a = Reverse(5);
        let b = Reverse(10);

        assert_eq!(a.partial_cmp(&b), Some(Ordering::Greater)); // 5 < 10, а значит a > b
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Less)); // 10 > 5, а значит b < a
    }
}
