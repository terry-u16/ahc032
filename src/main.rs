use std::ops::Index;

use ac_library::ModInt998244353;
use grid::{Coord, CoordDiff, Map2d};
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    init_map: Board,
    stamps: Vec<Stamp>,
}

impl Input {
    const N: usize = 9;
    const M: usize = 20;
    const K: usize = 81;

    fn read_input() -> Self {
        input! {
            n: usize,
            m: usize,
            _k: usize,
        }

        input! {
            raw_map: [[u32; n]; n]
        }

        let mut init_map = Map2d::new_with(ModInt998244353::new(0), n);

        for row in 0..n {
            for col in 0..n {
                let c = Coord::new(row, col);
                init_map[c] = ModInt998244353::new(raw_map[row][col]);
            }
        }

        let mut stamps = Vec::with_capacity(m);

        for _ in 0..m {
            input! {
                raw_stamp: [[u32; 3]; 3]
            }

            let mut stamp = Map2d::new_with(ModInt998244353::new(0), 3);

            for row in 0..3 {
                for col in 0..3 {
                    let c = Coord::new(row, col);
                    stamp[c] = ModInt998244353::new(raw_stamp[row][col]);
                }
            }

            stamps.push(Stamp::new(stamp));
        }

        Self {
            init_map: Board::new(init_map),
            stamps,
        }
    }
}

#[derive(Debug, Clone)]
struct Board {
    map: Map2d<ModInt998244353>,
}

impl Board {
    fn new(map: Map2d<ModInt998244353>) -> Self {
        Self { map }
    }

    fn calc_score(&self) -> i64 {
        let mut score = 0;

        for value in self.map.iter() {
            score += value.val() as i64;
        }

        score
    }

    fn stamp(&mut self, stamp: &Stamp, coord: Coord) {
        self.map[coord] += stamp.values[Coord::new(0, 0)];
        self.map[coord + CoordDiff::new(0, 1)] += stamp.values[Coord::new(0, 1)];
        self.map[coord + CoordDiff::new(0, 2)] += stamp.values[Coord::new(0, 2)];
        self.map[coord + CoordDiff::new(1, 0)] += stamp.values[Coord::new(1, 0)];
        self.map[coord + CoordDiff::new(1, 1)] += stamp.values[Coord::new(1, 1)];
        self.map[coord + CoordDiff::new(1, 2)] += stamp.values[Coord::new(1, 2)];
        self.map[coord + CoordDiff::new(2, 0)] += stamp.values[Coord::new(2, 0)];
        self.map[coord + CoordDiff::new(2, 1)] += stamp.values[Coord::new(2, 1)];
        self.map[coord + CoordDiff::new(2, 2)] += stamp.values[Coord::new(2, 2)];
    }

    fn revert(&mut self, stamp: &Stamp, coord: Coord) {
        self.map[coord] -= stamp.values[Coord::new(0, 0)];
        self.map[coord + CoordDiff::new(0, 1)] -= stamp.values[Coord::new(0, 1)];
        self.map[coord + CoordDiff::new(0, 2)] -= stamp.values[Coord::new(0, 2)];
        self.map[coord + CoordDiff::new(1, 0)] -= stamp.values[Coord::new(1, 0)];
        self.map[coord + CoordDiff::new(1, 1)] -= stamp.values[Coord::new(1, 1)];
        self.map[coord + CoordDiff::new(1, 2)] -= stamp.values[Coord::new(1, 2)];
        self.map[coord + CoordDiff::new(2, 0)] -= stamp.values[Coord::new(2, 0)];
        self.map[coord + CoordDiff::new(2, 1)] -= stamp.values[Coord::new(2, 1)];
        self.map[coord + CoordDiff::new(2, 2)] -= stamp.values[Coord::new(2, 2)];
    }
}

impl Index<Coord> for Board {
    type Output = ModInt998244353;

    fn index(&self, index: Coord) -> &Self::Output {
        &self.map[index]
    }
}

#[derive(Debug, Clone)]
struct Stamp {
    values: Map2d<ModInt998244353>,
}

impl Stamp {
    fn new(values: Map2d<ModInt998244353>) -> Self {
        Self { values }
    }
}

impl Index<Coord> for Stamp {
    type Output = ModInt998244353;

    fn index(&self, index: Coord) -> &Self::Output {
        &self.values[index]
    }
}

fn main() {
    let input = Input::read_input();
    let mut board = input.init_map.clone();
    let mut result = vec![];

    for row in 0..Input::N - 3 {
        for col in 0..Input::N - 3 {
            let coord = Coord::new(row, col);
            let mut best_pos = None;
            let mut best_score = board[coord].val();

            for stamp1 in 0..Input::M {
                let c = Coord::new(0, 0);
                let score = (board[coord] + input.stamps[stamp1][c]).val();
                if best_score.change_max(score) {
                    best_pos = Some((Coord::new(row, col), vec![stamp1]));
                }
            }

            if best_score < 950000000 {
                for stamp1 in 0..Input::M {
                    let c = Coord::new(0, 0);
                    let score = (board[coord] + input.stamps[stamp1][c]).val();
                    if best_score.change_max(score) {
                        best_pos = Some((Coord::new(row, col), vec![stamp1]));
                    }

                    for stamp2 in stamp1..Input::M {
                        let score =
                            (board[coord] + input.stamps[stamp1][c] + input.stamps[stamp2][c])
                                .val();
                        if best_score.change_max(score) {
                            best_pos = Some((Coord::new(row, col), vec![stamp1, stamp2]));
                        }
                    }
                }
            }

            if let Some((pos, stamp)) = best_pos {
                for s in stamp {
                    board.stamp(&input.stamps[s], pos);
                    result.push((pos, s));
                }
            }
        }
    }

    for _ in 0..Input::K {
        let mut best_pos = None;
        let mut best_score = board.calc_score();

        for row in 0..Input::N - 2 {
            for col in 0..Input::N - 2 {
                let c = Coord::new(row, col);

                if result.len() + 1 <= Input::K {
                    for t1 in 0..Input::M {
                        board.stamp(&input.stamps[t1], c);

                        if best_score.change_max(board.calc_score()) {
                            best_pos = Some((c, vec![t1]));
                        }

                        if result.len() + 2 <= Input::K {
                            for t2 in t1..Input::M {
                                board.stamp(&input.stamps[t2], c);

                                if best_score.change_max(board.calc_score()) {
                                    best_pos = Some((c, vec![t1, t2]));
                                }

                                if result.len() + 3 <= Input::K {
                                    for t3 in t2..Input::M {
                                        board.stamp(&input.stamps[t3], c);

                                        if best_score.change_max(board.calc_score()) {
                                            best_pos = Some((c, vec![t1, t2, t3]));
                                        }

                                        board.revert(&input.stamps[t3], c);
                                    }
                                }

                                board.revert(&input.stamps[t2], c);
                            }
                        }

                        board.revert(&input.stamps[t1], c);
                    }
                }
            }
        }

        if let Some((pos, stamp)) = best_pos {
            for &stamp in stamp.iter() {
                board.stamp(&input.stamps[stamp], pos);
                result.push((pos, stamp));
            }
        }
    }

    println!("{}", result.len());

    for &(pos, stamp) in result.iter() {
        println!("{} {} {}", stamp, pos.row, pos.col);
    }
}

mod grid {
    use std::{fmt::Display, vec};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
    pub struct Coord {
        pub row: usize,
        pub col: usize,
    }

    #[allow(dead_code)]
    impl Coord {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size && self.col < size
        }

        pub const fn to_index(&self, size: usize) -> CoordIndex {
            CoordIndex(self.row * size + self.col)
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: usize, x1: usize) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }
    }

    impl Display for Coord {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({}, {})", self.row, self.col)
        }
    }

    impl TryFrom<CoordDiff> for Coord {
        type Error = String;

        fn try_from(diff: CoordDiff) -> Result<Self, Self::Error> {
            if diff.dr < 0 || diff.dc < 0 {
                Err(format!("{} をCoordに変換できません", diff))
            } else {
                Ok(Self::new(diff.dr as usize, diff.dc as usize))
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
    pub struct CoordDiff {
        pub dr: isize,
        pub dc: isize,
    }

    #[allow(dead_code)]
    impl CoordDiff {
        pub const fn new(dr: isize, dc: isize) -> Self {
            Self { dr, dc }
        }

        pub const fn invert(&self) -> Self {
            Self::new(-self.dr, -self.dc)
        }
    }

    impl Display for CoordDiff {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({}, {})", self.dr, self.dc)
        }
    }

    impl From<Coord> for CoordDiff {
        fn from(coord: Coord) -> Self {
            Self::new(coord.row as isize, coord.col as isize)
        }
    }

    impl std::ops::Add<CoordDiff> for Coord {
        type Output = Coord;

        fn add(self, rhs: CoordDiff) -> Self::Output {
            Coord::new(
                self.row.wrapping_add_signed(rhs.dr),
                self.col.wrapping_add_signed(rhs.dc),
            )
        }
    }

    impl std::ops::Add<CoordDiff> for CoordDiff {
        type Output = CoordDiff;

        fn add(self, rhs: CoordDiff) -> Self::Output {
            CoordDiff::new(self.dr + rhs.dr, self.dc + rhs.dc)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct CoordIndex(pub usize);

    #[allow(dead_code)]
    impl CoordIndex {
        pub const fn new(index: usize) -> Self {
            Self(index)
        }

        pub const fn to_coord(&self, n: usize) -> Coord {
            Coord::new(self.0 / n, self.0 % n)
        }
    }

    #[allow(dead_code)]
    pub const ADJACENTS: [CoordDiff; 4] = [
        CoordDiff::new(!0, 0),
        CoordDiff::new(0, 1),
        CoordDiff::new(1, 0),
        CoordDiff::new(0, !0),
    ];

    #[allow(dead_code)]
    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub size: usize,
        map: Vec<T>,
    }

    #[allow(dead_code)]
    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, size: usize) -> Self {
            assert_eq!(size * size, map.len());
            Self { size, map }
        }

        pub fn iter(&self) -> impl Iterator<Item = &T> {
            self.map.iter()
        }

        pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
            self.map.iter_mut()
        }
    }

    #[allow(dead_code)]
    impl<T: Clone> Map2d<T> {
        pub fn new_with(v: T, size: usize) -> Self {
            let map = vec![v; size * size];
            Self::new(map, size)
        }
    }

    impl<T> std::ops::Index<Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self[coordinate.to_index(self.size)]
        }
    }

    impl<T> std::ops::IndexMut<Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            let size = self.size;
            &mut self[coordinate.to_index(size)]
        }
    }

    impl<T> std::ops::Index<CoordIndex> for Map2d<T> {
        type Output = T;

        fn index(&self, index: CoordIndex) -> &Self::Output {
            &self.map[index.0]
        }
    }

    impl<T> std::ops::IndexMut<CoordIndex> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
            &mut self.map[index.0]
        }
    }
}
