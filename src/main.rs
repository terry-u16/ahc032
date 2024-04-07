use std::{ops::Index, time::Instant, vec};

use ac_library::ModInt998244353;
use grid::{Coord, CoordDiff, Map2d};
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

use crate::beam::NoOpDeduplicator;

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
    _stamps: Vec<Stamp>,
    mul_stamps: Vec<Vec<Stamp>>,
    mul_stamp_raw: Vec<Vec<Vec<usize>>>,
    targets: Vec<Coord>,
    max_ops: Vec<usize>,
    since: Instant,
}

impl Input {
    const N: usize = 9;
    const M: usize = 20;
    const _K: usize = 81;

    fn read_input() -> Self {
        input! {
            n: usize,
            m: usize,
            _k: usize,
        }

        input! {
            raw_map: [[u32; n]; n]
        }

        let since = Instant::now();

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

        let mut mul_stamps = vec![];
        let mut mul_stamp_raw = vec![];

        for i in 0..=4 {
            let mut stamp_set = vec![];
            let mut stamp_set_raw = vec![];
            let mut stamp_raw = vec![];
            let mut new_stamp = Stamp::new(Map2d::new_with(ModInt998244353::new(0), 3));

            Self::stamp_dfs(
                &stamps,
                &mut stamp_raw,
                &mut new_stamp,
                &mut stamp_set,
                &mut stamp_set_raw,
                0,
                0,
                i,
            );

            mul_stamps.push(stamp_set);
            mul_stamp_raw.push(stamp_set_raw);
        }

        let mut targets = vec![];
        let mut max_ops = vec![];
        let mut max_turn = 0.0f64;

        fn get_turn(row: usize, col: usize) -> f64 {
            if row == 6 && col == 6 {
                3.0
            } else if row == 6 || col == 6 {
                2.0
            } else {
                1.5
            }
        }

        for pivot in 0..Input::N - 2 {
            for row in pivot..Input::N - 2 {
                targets.push(Coord::new(row, pivot));
                max_turn += get_turn(row, pivot);
                max_ops.push(max_turn.ceil() as usize);
            }

            for col in pivot + 1..Input::N - 2 {
                targets.push(Coord::new(pivot, col));
                max_turn += get_turn(pivot, col);
                max_ops.push(max_turn.ceil() as usize);
            }
        }

        Self {
            init_map: Board::new(init_map),
            _stamps: stamps,
            mul_stamps,
            mul_stamp_raw,
            targets,
            max_ops,
            since,
        }
    }

    fn stamp_dfs(
        stamps: &[Stamp],
        stamp_raw: &mut Vec<usize>,
        current: &mut Stamp,
        stamp_set: &mut Vec<Stamp>,
        stamp_set_raw: &mut Vec<Vec<usize>>,
        first_index: usize,
        depth: usize,
        max_depth: usize,
    ) {
        if depth == max_depth {
            stamp_set.push(current.clone());
            stamp_set_raw.push(stamp_raw.clone());
            return;
        }

        for s in first_index..Input::M {
            let stamp = &stamps[s];
            stamp_raw.push(s);

            for (source, dst) in stamp.values.iter().zip(current.values.iter_mut()) {
                *dst += *source;
            }

            Self::stamp_dfs(
                stamps,
                stamp_raw,
                current,
                stamp_set,
                stamp_set_raw,
                s,
                depth + 1,
                max_depth,
            );

            stamp_raw.pop();

            for (source, dst) in stamp.values.iter().zip(current.values.iter_mut()) {
                *dst -= *source;
            }
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

struct LargeState {
    board: Board,
    input: Input,
    turn: usize,
    score: i64,
}

impl LargeState {
    fn new(board: Board, input: Input, turn: usize, score: i64) -> Self {
        Self {
            board,
            input,
            turn,
            score,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SmallState {
    score: i64,
    prev_score: i64,
    stamp_count: usize,
    action: (Coord, usize, usize),
}

impl beam::SmallState for SmallState {
    type Score = i64;

    type Hash = u64;

    type LargeState = LargeState;

    type Action = (Coord, usize, usize);

    fn raw_score(&self) -> Self::Score {
        self.score
    }

    fn hash(&self) -> Self::Hash {
        0
    }

    fn apply(&self, state: &mut Self::LargeState) {
        let (pos, stamp0, stamp1) = self.action;
        state
            .board
            .stamp(&state.input.mul_stamps[stamp0][stamp1], pos);

        state.turn += 1;
        state.score = self.score;
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        let (pos, stamp0, stamp1) = self.action;
        state
            .board
            .revert(&state.input.mul_stamps[stamp0][stamp1], pos);

        state.turn -= 1;
        state.score = self.prev_score;
    }

    fn action(&self) -> Self::Action {
        self.action
    }
}

struct ActionGenerator {
    input: Input,
}

impl beam::ActGen<SmallState> for ActionGenerator {
    fn generate(
        &self,
        small_state: &SmallState,
        large_state: &<SmallState as beam::SmallState>::LargeState,
        next_states: &mut Vec<SmallState>,
    ) {
        let turn = large_state.turn;
        let prev_score = large_state.score;
        let remaining_ops = (large_state.input.max_ops[turn] - small_state.stamp_count).min(2);
        let coord = self.input.targets[turn];

        if coord.row == 6 && coord.col == 6 {
            let remaining_ops = (large_state.input.max_ops[turn] - small_state.stamp_count).min(3);
            const CDS: [CoordDiff; 9] = [
                CoordDiff::new(0, 0),
                CoordDiff::new(0, 1),
                CoordDiff::new(0, 2),
                CoordDiff::new(1, 0),
                CoordDiff::new(1, 1),
                CoordDiff::new(1, 2),
                CoordDiff::new(2, 0),
                CoordDiff::new(2, 1),
                CoordDiff::new(2, 2),
            ];

            let old_v = CDS.map(|cd| large_state.board.map[coord + cd]);

            for cnt in 0..=remaining_ops {
                for (j, stamp) in self.input.mul_stamps[cnt].iter().enumerate() {
                    let mut sum = 0;

                    for (src, tgt) in old_v.iter().zip(stamp.values.iter()) {
                        sum += (src + tgt).val() as i64;
                    }

                    let score = prev_score + sum;

                    let new_state = SmallState {
                        score,
                        prev_score,
                        stamp_count: small_state.stamp_count + cnt,
                        action: (coord, cnt, j),
                    };

                    next_states.push(new_state);
                }
            }
        } else if coord.row == 6 {
            let remaining_ops = (large_state.input.max_ops[turn] - small_state.stamp_count).min(3);

            const CD1: CoordDiff = CoordDiff::new(1, 0);
            const CD2: CoordDiff = CoordDiff::new(2, 0);
            const C0: Coord = Coord::new(0, 0);
            const C1: Coord = Coord::new(1, 0);
            const C2: Coord = Coord::new(2, 0);

            let v0 = large_state.board.map[coord];
            let v1 = large_state.board.map[coord + CD1];
            let v2 = large_state.board.map[coord + CD2];

            for cnt in 0..=remaining_ops {
                for (j, stamp) in self.input.mul_stamps[cnt].iter().enumerate() {
                    let mut sum = 0;
                    sum += (v0 + stamp[C0]).val() as i64;
                    sum += (v1 + stamp[C1]).val() as i64;
                    sum += (v2 + stamp[C2]).val() as i64;

                    let score = prev_score + sum;

                    let new_state = SmallState {
                        score,
                        prev_score,
                        stamp_count: small_state.stamp_count + cnt,
                        action: (coord, cnt, j),
                    };

                    next_states.push(new_state);
                }
            }
        } else if coord.col == 6 {
            let remaining_ops = (large_state.input.max_ops[turn] - small_state.stamp_count).min(3);

            const CD1: CoordDiff = CoordDiff::new(0, 1);
            const CD2: CoordDiff = CoordDiff::new(0, 2);
            const C0: Coord = Coord::new(0, 0);
            const C1: Coord = Coord::new(0, 1);
            const C2: Coord = Coord::new(0, 2);

            let v0 = large_state.board.map[coord];
            let v1 = large_state.board.map[coord + CD1];
            let v2 = large_state.board.map[coord + CD2];

            for cnt in 0..=remaining_ops {
                for (j, stamp) in self.input.mul_stamps[cnt].iter().enumerate() {
                    let mut sum = 0;

                    sum += (v0 + stamp[C0]).val() as i64;
                    sum += (v1 + stamp[C1]).val() as i64;
                    sum += (v2 + stamp[C2]).val() as i64;

                    let score = prev_score + sum;

                    let new_state = SmallState {
                        score,
                        prev_score,
                        stamp_count: small_state.stamp_count + cnt,
                        action: (coord, cnt, j),
                    };

                    next_states.push(new_state);
                }
            }
        } else {
            const C0: Coord = Coord::new(0, 0);
            let v0 = large_state.board.map[coord];

            for cnt in 0..=remaining_ops {
                for (j, stamp) in self.input.mul_stamps[cnt].iter().enumerate() {
                    let mut sum = 0;
                    sum += (v0 + stamp[C0]).val() as i64;

                    let score = prev_score + sum;

                    let new_state = SmallState {
                        score,
                        prev_score,
                        stamp_count: small_state.stamp_count + cnt,
                        action: (coord, cnt, j),
                    };

                    next_states.push(new_state);
                }
            }
        }
    }
}

fn main() {
    let input = Input::read_input();
    let board = input.init_map.clone();

    let large_state = LargeState::new(board, input.clone(), 0, 0);
    let small_state = SmallState::default();
    let action_generator = ActionGenerator {
        input: input.clone(),
    };
    let mut beam = beam::BeamSearch::new(large_state, small_state, action_generator);

    let deduplicator = NoOpDeduplicator;
    let beam_width = beam::BayesianBeamWidthSuggester::new(49, 1, 1.98, 3000, 100, 10000, 1);
    let (actions, _) = beam.run(49, beam_width, deduplicator);

    let mut result = vec![];

    for (pos, i, j) in actions {
        for &index in input.mul_stamp_raw[i][j].iter() {
            result.push((pos, index));
        }
    }

    println!("{}", result.len());

    for &(pos, stamp) in result.iter() {
        println!("{} {} {}", stamp, pos.row, pos.col);
    }

    eprintln!("Elapsed: {:?}", input.since.elapsed());
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

#[allow(dead_code)]
mod beam {
    //! ビームサーチライブラリ
    //! [rhooさんの記事](https://qiita.com/rhoo/items/2f647e32f6ff2c6ee056)を大いに参考にさせて頂きました。
    //! ありがとうございます……。
    //!
    //! # Usage
    //!
    //! ```
    //!
    //! // 状態のうち、差分計算を行わない部分
    //! struct SmallState;
    //!
    //! impl beam::SmallState for SmallState {
    //!     type Score = i64;
    //!     type Hash = u64;
    //!     type LargeState = LargeState;
    //!     type Action = usize;
    //!
    //!     // 略
    //! }
    //!
    //! // 状態のうち、差分計算を行う部分
    //! struct LargeState;
    //!
    //! // 次の行動を生成する構造体
    //! struct ActionGenerator;
    //!
    //! impl ActGen<SmallState> for ActionGenerator {
    //!     // 略
    //! }
    //!
    //! fn beam() -> Vec<usize> {
    //!     let large_state = LargeState;
    //!     let small_state = SmallState;
    //!     let action_generator = ActionGenerator;
    //!     let mut beam = BeamSearch::new(large_state, small_state, action_generator);
    //!
    //!     let deduplicator = NoOpDeduplicator;
    //!     let beam_width = FixedBeamWidthSuggester::new(100);
    //!     let (actions, score) = beam.run(2500, beam_width, deduplicator);
    //!
    //!     actions
    //! }
    //!
    //! ```
    //!
    use std::{
        cmp::Reverse,
        fmt::Display,
        hash::Hash,
        ops::{Index, IndexMut},
        time::Instant,
    };

    use rustc_hash::FxHashSet;

    use crate::bayesian::GaussInverseGamma;

    /// コピー可能な小さい状態を表すトレイト
    pub trait SmallState {
        type Score: Ord + Display;
        type Hash: Hash + Eq;
        type LargeState;
        type Action;

        /// ビームサーチ用スコア（大きいほど良い）
        /// デフォルトでは生スコアをそのまま返す
        fn beam_score(&self) -> Self::Score {
            self.raw_score()
        }

        // 生スコア
        fn raw_score(&self) -> Self::Score;

        /// ハッシュ値
        fn hash(&self) -> Self::Hash;

        /// stateにこの差分を作用させる
        fn apply(&self, state: &mut Self::LargeState);

        /// stateに作用させたこの差分をロールバックする
        fn rollback(&self, state: &mut Self::LargeState);

        /// 実行した行動を返す
        fn action(&self) -> Self::Action;
    }

    /// 現在のstateからの遷移先を列挙するトレイト
    pub trait ActGen<S: SmallState> {
        /// 現在のstateからの遷移先をnext_satesに格納する
        fn generate(&self, small_state: &S, large_state: &S::LargeState, next_states: &mut Vec<S>);
    }

    /// ビームの次の遷移候補
    struct Cancidate<S: SmallState> {
        /// 実行後のsmall_state
        small_state: S,
        /// 親となるノードのインデックス
        parent: NodeIndex,
    }

    impl<S: SmallState> Cancidate<S> {
        fn new(small_state: S, parent: NodeIndex) -> Self {
            Self {
                small_state,
                parent,
            }
        }

        fn to_node(
            self,
            child: NodeIndex,
            left_sibling: NodeIndex,
            right_sibling: NodeIndex,
        ) -> Node<S> {
            Node {
                small_state: self.small_state,
                parent: self.parent,
                child,
                left_sibling,
                right_sibling,
            }
        }
    }

    /// 重複除去を行うトレイト
    pub trait Deduplicator<S: SmallState> {
        /// 重複除去に使った情報をクリアし、次の重複除去の準備をする
        fn clear(&mut self);

        /// 重複チェックを行い、残すべきならtrue、重複していればfalseを返す
        fn filter(&mut self, state: &S) -> bool;
    }

    /// 重複除去を行わず素通しするDeduplicator
    pub struct NoOpDeduplicator;

    impl<S: SmallState> Deduplicator<S> for NoOpDeduplicator {
        fn clear(&mut self) {
            // do nothing
        }

        fn filter(&mut self, _state: &S) -> bool {
            // 常に素通しする
            true
        }
    }

    /// 同じハッシュ値を持つ状態を1つだけに制限するDeduplicator
    pub struct HashSingleDeduplicator<S: SmallState> {
        set: FxHashSet<S::Hash>,
    }

    impl<S: SmallState> HashSingleDeduplicator<S> {
        pub fn new() -> Self {
            Self {
                set: FxHashSet::default(),
            }
        }
    }

    impl<S: SmallState> Deduplicator<S> for HashSingleDeduplicator<S> {
        fn clear(&mut self) {
            self.set.clear();
        }

        fn filter(&mut self, state: &S) -> bool {
            // ハッシュが重複していなければ通す
            self.set.insert(state.hash())
        }
    }

    /// ビーム幅を提案するトレイト
    pub trait BeamWidthSuggester {
        // 現在のターン数を受け取り、ビーム幅を提案する
        fn suggest(&mut self) -> usize;
    }

    /// 常に固定のビーム幅を返すBeamWidthSuggester
    pub struct FixedBeamWidthSuggester {
        width: usize,
    }

    impl FixedBeamWidthSuggester {
        pub fn new(width: usize) -> Self {
            Self { width }
        }
    }

    impl BeamWidthSuggester for FixedBeamWidthSuggester {
        fn suggest(&mut self) -> usize {
            self.width
        }
    }

    /// ベイズ推定により適切なビーム幅を計算するBeamWidthSuggester。
    /// 1ターンあたりの実行時間が正規分布に従うと仮定し、+3σ分の余裕を持ってビーム幅を決める。
    ///
    /// ## モデル
    ///
    /// カルマンフィルタを適用するにあたって、以下のモデルを考える。
    ///
    /// - `i` ターン目のビーム幅1あたりの所要時間の平均値 `t_i` が正規分布 `N(μ_i, σ_i^2)` に従うと仮定する。
    ///   - 各ターンに観測される所要時間が `N(μ_i, σ_i^2)` に従うのではなく、所要時間の**平均値**が `N(μ_i, σ_i^2)` に従うとしている点に注意。
    ///     - すなわち `μ_i` は所要時間の平均値の平均値であり、所要時間の平均値が `μ_i` を中心とした確率分布を形成しているものとしている。ややこしい。
    ///   - この `μ_i` , `σ_i^2` をベイズ推定によって求めたい。
    /// - 所要時間 `t_i` は `t_{i+1}=t_i+N(0, α^2)` により更新されるものとする。
    ///   - `N(0, α^2)` は標準偏差 `α` のノイズを意味する。お気持ちとしては「実行時間がターン経過に伴ってちょっとずつ変わっていくことがあるよ」という感じ。
    ///   - `α` は既知の定数とし、適当に決める。
    ///   - 本来は問題に合わせたちゃんとした更新式にすべき（ターン経過に伴って線形に増加するなど）なのだが、事前情報がないため大胆に仮定する。
    /// - 所要時間の観測値 `τ_i` は `τ_i=t_i+N(0, β^2)` により得られるものとする。
    ///   - `β` は既知の定数とし、適当に決める。
    ///   - 本来この `β` も推定できると嬉しいのだが、取扱いが煩雑になるためこちらも大胆に仮定する。
    ///
    /// ## モデルの初期化
    ///
    /// - `μ_0` は実行時間制限を `T` 、標準ビーム幅を `W` 、実行ターン数を `M` として、 `μ_0=T/WM` などとすればよい。
    /// - `σ_0` は適当に `σ_0=0.1μ_0` とする。ここは標準ビーム幅にどのくらい自信があるかによる。
    /// - `α` は適当に `α=0.01μ_0` とする。定数は本当に勘。多分問題に合わせてちゃんと考えた方が良い。
    /// - `β` は `σ_0=0.05μ_0` とする。適当なベンチマーク問題で標準偏差を取ったらそのくらいだったため。
    ///
    /// ## モデルの更新
    ///
    /// 以下のように更新をかけていく。
    ///
    /// 1. `t_0=N(μ_0, σ_0^2)` と初期化する。
    /// 2. `t_1=t_0+N(0, α^2)` とし、事前分布 `t_1=N(μ_1, σ_1^2)=N(μ_0, σ_0^2+α^2)` を得る。ここはベイズ更新ではなく単純な正規分布の合成でよい。
    /// 3. `τ_1` が観測されるので、ベイズ更新して事後分布 `N(μ_1', σ_1^2')` を得る。
    /// 4. 同様に `t_2=N(μ_2, σ_2^2)` を得る。
    /// 5. `τ_2` を用いてベイズ更新。以下同様。
    ///
    /// ## 適切なビーム幅の推定
    ///
    /// - 余裕を持って、99.8%程度の確率（+3σ）で実行時間制限に収まるようなビーム幅にしたい。
    /// - ここで、 `t_i=t_{i+1}=･･･=t_M=N(μ_i, σ_i^2)` と大胆仮定する。
    ///   - `α` によって `t_i` がどんどん変わってしまうと考えるのは保守的すぎるため。
    /// - すると残りターン数 `M_i=M-i` として、 `Στ_i=N(M_i*μ_i, M_i*σ_i^2)` となる。
    /// - したがって、残り時間を `T_i` として `W(M_i*μ_i+3(σ_i√M_i))≦T_i` となる最大の `W` を求めればよく、 `W=floor(T_i/(M_i*μ_i+3(σ_i√M_i)))` となる。
    /// - 最後に、念のため適当な `W_min` , `W_max` でclampしておく。
    pub struct BayesianBeamWidthSuggester {
        /// ターンごとの所要時間が従う正規分布の(平均, 標準偏差)の事前分布
        prior_dist: GaussInverseGamma,
        /// 問題の実行時間制限T
        time_limit_sec: f64,
        /// 現在のターン数i
        current_turn: usize,
        /// 最大ターン数M
        max_turn: usize,
        /// ウォームアップターン数（最初のXターン分の情報は採用せずに捨てる）
        warmup_turn: usize,
        /// 所要時間を記憶するターン数の目安
        max_memory_turn: usize,
        /// 最小ビーム幅W_min
        min_beam_width: usize,
        /// 最大ビーム幅W_max
        max_beam_width: usize,
        /// 現在のビーム幅W_i
        current_beam_width: usize,
        /// ログの出力インターバル（0にするとログを出力しなくなる）
        verbose_interval: usize,
        /// ビーム開始時刻
        start_time: Instant,
        /// 前回の計測時刻
        last_time: Instant,
    }

    impl BayesianBeamWidthSuggester {
        pub fn new(
            max_turn: usize,
            warmup_turn: usize,
            time_limit_sec: f64,
            standard_beam_width: usize,
            min_beam_width: usize,
            max_beam_width: usize,
            verbose_interval: usize,
        ) -> Self {
            assert!(
                max_turn * standard_beam_width > 0,
                "ターン数とビーム幅設定が不正です。"
            );
            assert!(
                min_beam_width > 0,
                "最小のビーム幅は正の値でなければなりません。"
            );
            assert!(
                min_beam_width <= max_beam_width,
                "最大のビーム幅は最小のビーム幅以上でなければなりません。"
            );

            let mean_sec = time_limit_sec / (max_turn * standard_beam_width) as f64;

            // 雑にσ=20%ズレると仮定
            let stddev_sec = 0.2 * mean_sec;
            let prior_dist = GaussInverseGamma::from_psuedo_observation(mean_sec, stddev_sec, 3);

            // 直近20%程度のターン数の移動平均的な所要時間を参考にする
            let max_memory_turn = max_turn / 5;

            eprintln!(
                "standard beam width: {}, time limit: {:.3}s",
                standard_beam_width, time_limit_sec
            );

            Self {
                prior_dist,
                time_limit_sec,
                current_turn: 0,
                min_beam_width,
                max_beam_width,
                verbose_interval,
                max_turn,
                max_memory_turn,
                warmup_turn,
                current_beam_width: 0,
                start_time: Instant::now(),
                last_time: Instant::now(),
            }
        }

        fn update_distribution(&mut self, duration_sec: f64) {
            self.prior_dist.update(duration_sec);

            // ベイズ推定の疑似観測数にリミットをかける
            // （序盤と終盤で実行時間が異なるケースで、序盤の観測値に引きずられないようにするため）
            if self.prior_dist.get_pseudo_observation_count() >= self.max_memory_turn as f64 {
                self.prior_dist
                    .set_pseudo_observation_count(self.max_memory_turn as f64);
            }
        }

        fn calc_safe_beam_width(&self) -> usize {
            let remaining_turn = (self.max_turn - self.current_turn) as f64;
            let elapsed_time = (Instant::now() - self.start_time).as_secs_f64();
            let remaining_time = self.time_limit_sec - elapsed_time;

            let (mean, std_dev) = self.prior_dist.expected();
            let variance = std_dev * std_dev;

            let mean_remaining = remaining_turn * mean;
            let variance_remaining = remaining_turn * variance;
            let std_dev_remaining = variance_remaining.sqrt();

            // 2σの余裕を持たせる
            const SIGMA_COEF: f64 = 2.0;
            let needed_time_per_width = mean_remaining + SIGMA_COEF * std_dev_remaining;
            let beam_width = ((remaining_time / needed_time_per_width) as usize)
                .clamp(self.min_beam_width, self.max_beam_width);

            if self.verbose_interval != 0 && self.current_turn % self.verbose_interval == 0 {
                let stddev_per_run = (self.max_turn as f64 * variance).sqrt();
                let stddev_per_turn = variance.sqrt();

                eprintln!(
                    "turn:{:5}, beam width:{:5}, pase:{:7.1} ±{:6.2}ms/run,{:6.3} ±{:6.3}ms/turn",
                    self.current_turn,
                    beam_width,
                    mean * (beam_width * self.max_turn) as f64 * 1e3,
                    stddev_per_run * beam_width as f64 * 1e3,
                    mean * beam_width as f64 * 1e3,
                    stddev_per_turn * beam_width as f64 * 1e3
                );
            }

            beam_width
        }
    }

    impl BeamWidthSuggester for BayesianBeamWidthSuggester {
        fn suggest(&mut self) -> usize {
            assert!(
                self.current_turn < self.max_turn,
                "規定ターン終了後にsuggest()が呼び出されました。"
            );

            if self.current_turn >= self.warmup_turn {
                let elapsed = (Instant::now() - self.last_time).as_secs_f64();
                let elapsed_per_beam = elapsed / self.current_beam_width as f64;
                self.update_distribution(elapsed_per_beam);
            }

            self.last_time = Instant::now();
            let beam_width = self.calc_safe_beam_width();
            self.current_beam_width = beam_width;
            self.current_turn += 1;
            beam_width
        }
    }

    /// ビームサーチ木のノード
    #[derive(Debug, Default, Clone)]
    struct Node<S: SmallState> {
        /// 実行後のsmall_state
        small_state: S,
        /// （N分木と考えたときの）親ノード
        parent: NodeIndex,
        /// （二重連鎖木と考えたときの）子ノード
        child: NodeIndex,
        /// （二重連鎖木と考えたときの）左の兄弟ノード
        left_sibling: NodeIndex,
        /// （二重連鎖木と考えたときの）右の兄弟ノード
        right_sibling: NodeIndex,
    }

    impl<S: SmallState> Node<S> {
        fn new(
            small_state: S,
            parent: NodeIndex,
            child: NodeIndex,
            left_sibling: NodeIndex,
            right_sibling: NodeIndex,
        ) -> Self {
            Self {
                small_state,
                parent,
                child,
                left_sibling,
                right_sibling,
            }
        }
    }

    /// NodeVec用のindex
    /// 型安全性と、indexの内部的な型(u32 or u16)の変更を容易にすることが目的
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct NodeIndex(u16);

    impl NodeIndex {
        /// 何も指していないことを表す定数
        const NULL: NodeIndex = NodeIndex(!0);
    }

    impl Default for NodeIndex {
        fn default() -> Self {
            Self::NULL
        }
    }

    impl From<usize> for NodeIndex {
        fn from(value: usize) -> Self {
            Self(value as u16)
        }
    }

    impl Into<usize> for NodeIndex {
        fn into(self) -> usize {
            self.0 as usize
        }
    }

    /// Nodeのコレクション
    #[derive(Debug)]
    struct NodeVec<S: SmallState> {
        nodes: Vec<Node<S>>,
        free_indices: Vec<usize>,
    }

    impl<S: SmallState + Default + Clone> NodeVec<S> {
        fn new(capacity: usize) -> Self {
            Self {
                nodes: vec![Default::default(); capacity],
                free_indices: (0..capacity).rev().collect(),
            }
        }

        fn push(&mut self, node: Node<S>) -> NodeIndex {
            let index = self
                .free_indices
                .pop()
                .expect("ノードプールの容量制限に達しました。");

            self.nodes[index] = node;

            NodeIndex::from(index)
        }

        fn delete(&mut self, index: NodeIndex) {
            self.free_indices.push(index.into());
        }
    }

    impl<S: SmallState> Index<NodeIndex> for NodeVec<S> {
        type Output = Node<S>;

        fn index(&self, index: NodeIndex) -> &Self::Output {
            let index: usize = index.into();
            self.nodes.index(index)
        }
    }

    impl<S: SmallState> IndexMut<NodeIndex> for NodeVec<S> {
        fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
            let index: usize = index.into();
            self.nodes.index_mut(index)
        }
    }

    /// 保持する最大ノード数。65536個にするとNULLノードと被るため65535個に抑えている
    const MAX_NODES: usize = std::u16::MAX as usize - 1;

    #[derive(Debug)]
    pub struct BeamSearch<S: SmallState, G: ActGen<S>> {
        state: S::LargeState,
        act_gen: G,
        nodes: NodeVec<S>,
        current_index: NodeIndex,
        leaves: Vec<NodeIndex>,
        next_leaves: Vec<NodeIndex>,
        action_buffer: Vec<S>,
    }

    impl<S: SmallState + Default + Clone, G: ActGen<S>> BeamSearch<S, G> {
        /// ビーム木を指定された容量で初期化する
        pub fn new(large_state: S::LargeState, small_state: S, act_gen: G) -> Self {
            let mut nodes = NodeVec::new(MAX_NODES);
            nodes.push(Node::new(
                small_state,
                NodeIndex::NULL,
                NodeIndex::NULL,
                NodeIndex::NULL,
                NodeIndex::NULL,
            ));

            Self {
                state: large_state,
                act_gen,
                nodes,
                current_index: NodeIndex(0),
                leaves: vec![NodeIndex(0)],
                next_leaves: vec![],
                action_buffer: vec![],
            }
        }

        pub fn run<W: BeamWidthSuggester, P: Deduplicator<S>>(
            &mut self,
            max_turn: usize,
            mut beam_width_suggester: W,
            mut deduplicator: P,
        ) -> (Vec<S::Action>, S::Score) {
            let mut candidates = vec![];

            for turn in 0..max_turn {
                let beam_width = beam_width_suggester.suggest();
                candidates.clear();
                self.dfs(&mut candidates, true);

                if turn + 1 == max_turn {
                    break;
                }

                assert_ne!(
                    candidates.len(),
                    0,
                    "次の状態の候補が見つかりませんでした。"
                );

                // 重複除去を行ったのち、次の遷移先を確定させる
                // glidesortが速いらしいが、多様性を確保したいため敢えて不安定ソートを採用している
                candidates.sort_unstable_by_key(|c| Reverse(c.small_state.beam_score()));

                deduplicator.clear();
                self.update_tree(
                    candidates
                        .drain(..)
                        .filter(|c| deduplicator.filter(&c.small_state))
                        .take(beam_width),
                );
            }

            let Cancidate {
                small_state,
                parent,
                ..
            } = candidates
                .into_iter()
                .max_by_key(|c| c.small_state.beam_score())
                .expect("最終状態となる候補が見つかりませんでした。");

            // 操作列の復元
            let mut actions = self.restore_actions(parent);
            actions.push(small_state.action());
            (actions, small_state.raw_score())
        }

        /// ノードを追加する
        fn add_node(&mut self, candidate: Cancidate<S>) {
            let parent = candidate.parent;
            let node_index = self.nodes.push(candidate.to_node(
                NodeIndex::NULL,
                NodeIndex::NULL,
                NodeIndex::NULL,
            ));

            // 親の子、すなわち一番左にいる兄弟ノード
            let sibling = self.nodes[parent].child;

            // 既に兄弟がいる場合、その左側に入る
            if sibling != NodeIndex::NULL {
                self.nodes[sibling].left_sibling = node_index;
            }

            // 兄弟を1マス右に押し出して、自分が一番左に入る
            self.next_leaves.push(node_index);
            self.nodes[parent].child = node_index;
            self.nodes[node_index].right_sibling = sibling;
        }

        /// 指定されたインデックスのノードを削除する
        /// 必要に応じてビーム木の辺を繋ぎ直す
        fn remove_node(&mut self, mut index: NodeIndex) {
            loop {
                let Node {
                    left_sibling,
                    right_sibling,
                    parent,
                    ..
                } = self.nodes[index];
                self.nodes.delete(index);

                // 親は生きているはず
                assert_ne!(parent, NodeIndex::NULL, "rootノードを消そうとしています。");

                // もう兄弟がいなければ親へ
                if left_sibling == NodeIndex::NULL && right_sibling == NodeIndex::NULL {
                    index = parent;
                    continue;
                }

                // 左右の連結リストを繋ぎ直す
                if left_sibling != NodeIndex::NULL {
                    self.nodes[left_sibling].right_sibling = right_sibling;
                } else {
                    self.nodes[parent].child = right_sibling;
                }

                if right_sibling != NodeIndex::NULL {
                    self.nodes[right_sibling].left_sibling = left_sibling;
                }

                return;
            }
        }

        /// DFSでビームサーチ木を走査し、次の状態の一覧をcandidatesに詰める
        /// ビームサーチ木が一本道の場合は戻る必要がないため、is_single_pathで管理
        fn dfs(&mut self, candidates: &mut Vec<Cancidate<S>>, is_single_path: bool) {
            // 葉ノードであれば次の遷移を行う
            if self.nodes[self.current_index].child == NodeIndex::NULL {
                self.act_gen.generate(
                    &self.nodes[self.current_index].small_state,
                    &self.state,
                    &mut self.action_buffer,
                );

                while let Some(state) = self.action_buffer.pop() {
                    candidates.push(Cancidate::new(state, self.current_index));
                }

                return;
            }

            let current_index = self.current_index;
            let mut child_index = self.nodes[current_index].child;
            let next_is_single_path =
                is_single_path & (self.nodes[child_index].right_sibling == NodeIndex::NULL);

            // デバッグ用
            //let prev_state = self.state.clone();

            // 兄弟ノードを全て走査する
            loop {
                self.current_index = child_index;
                self.nodes[child_index].small_state.apply(&mut self.state);
                self.dfs(candidates, next_is_single_path);

                if !next_is_single_path {
                    self.nodes[child_index]
                        .small_state
                        .rollback(&mut self.state);

                    // デバッグ用
                    //assert!(prev_state == self.state);
                }

                child_index = self.nodes[child_index].right_sibling;

                if child_index == NodeIndex::NULL {
                    break;
                }
            }

            if !next_is_single_path {
                self.current_index = current_index;
            }
        }

        /// 木を更新する
        /// 具体的には以下の処理を行う
        ///
        /// - 新しいcandidatesを葉に追加する
        /// - 1ターン前のノードであって葉のノード（今後参照されないノード）を削除する
        fn update_tree(&mut self, candidates: impl Iterator<Item = Cancidate<S>>) {
            self.next_leaves.clear();
            for candidate in candidates {
                self.add_node(candidate);
            }

            for i in 0..self.leaves.len() {
                let node_index = self.leaves[i];

                if self.nodes[node_index].child == NodeIndex::NULL {
                    self.remove_node(node_index);
                }
            }

            std::mem::swap(&mut self.leaves, &mut self.next_leaves);
        }

        /// 操作列を復元する
        fn restore_actions(&self, mut index: NodeIndex) -> Vec<S::Action> {
            let mut actions = vec![];

            while self.nodes[index].parent != NodeIndex::NULL {
                actions.push(self.nodes[index].small_state.action());
                index = self.nodes[index].parent;
            }

            actions.reverse();
            actions
        }
    }

    #[cfg(test)]
    mod test {
        //! TSPをビームサーチで解くテスト
        use super::{ActGen, BeamSearch, FixedBeamWidthSuggester, NoOpDeduplicator};

        #[derive(Debug, Clone)]
        struct Input {
            n: usize,
            distances: Vec<Vec<i32>>,
        }

        impl Input {
            fn gen_testcase() -> Self {
                let n = 4;
                let distances = vec![
                    vec![0, 2, 3, 10],
                    vec![2, 0, 1, 3],
                    vec![3, 1, 0, 2],
                    vec![10, 3, 2, 0],
                ];

                Self { n, distances }
            }
        }

        #[derive(Debug, Clone, Copy)]
        struct SmallState {
            distance: i32,
            position: usize,
            visited_count: usize,
        }

        impl SmallState {
            fn new(distance: i32, position: usize, visited_count: usize) -> Self {
                Self {
                    distance,
                    position,
                    visited_count,
                }
            }
        }

        impl Default for SmallState {
            fn default() -> Self {
                Self {
                    distance: 0,
                    position: 0,
                    visited_count: 1,
                }
            }
        }

        impl super::SmallState for SmallState {
            type Score = i32;
            type Hash = u64;
            type LargeState = LargeState;
            type Action = usize;

            fn raw_score(&self) -> Self::Score {
                self.distance
            }

            fn beam_score(&self) -> Self::Score {
                // 大きいほど良いとする
                -self.distance
            }

            fn hash(&self) -> Self::Hash {
                // 適当に0を返す
                0
            }

            fn apply(&self, state: &mut Self::LargeState) {
                // 現在地を訪問済みにする
                state.visited[self.position] = true;
            }

            fn rollback(&self, state: &mut Self::LargeState) {
                // 現在地を未訪問にする
                state.visited[self.position] = false;
            }

            fn action(&self) -> Self::Action {
                self.position
            }
        }

        #[derive(Debug, Clone)]
        struct LargeState {
            visited: Vec<bool>,
        }

        impl LargeState {
            fn new(n: usize) -> Self {
                let mut visited = vec![false; n];
                visited[0] = true;
                Self { visited }
            }
        }

        #[derive(Debug, Clone)]
        struct ActionGenerator<'a> {
            input: &'a Input,
        }

        impl<'a> ActionGenerator<'a> {
            fn new(input: &'a Input) -> Self {
                Self { input }
            }
        }

        impl<'a> ActGen<SmallState> for ActionGenerator<'a> {
            fn generate(
                &self,
                small_state: &SmallState,
                large_state: &LargeState,
                next_states: &mut Vec<SmallState>,
            ) {
                if small_state.visited_count == self.input.n {
                    // 頂点0に戻るしかない
                    let next_pos = 0;
                    let next_dist =
                        small_state.distance + self.input.distances[small_state.position][0];
                    let next_visited_count = small_state.visited_count + 1;
                    let next_state = SmallState::new(next_dist, next_pos, next_visited_count);
                    next_states.push(next_state);
                    return;
                }

                // 未訪問の頂点に移動
                for next_pos in 0..self.input.n {
                    if large_state.visited[next_pos] {
                        continue;
                    }

                    let next_dist =
                        small_state.distance + self.input.distances[small_state.position][next_pos];
                    let next_visited_count = small_state.visited_count + 1;
                    let next_state = SmallState::new(next_dist, next_pos, next_visited_count);
                    next_states.push(next_state);
                }
            }
        }

        #[test]
        fn beam_tsp_test() {
            let input = Input::gen_testcase();
            let small_state = SmallState::default();
            let large_state = LargeState::new(input.n);
            let action_generator = ActionGenerator::new(&input);
            let mut beam = BeamSearch::new(large_state, small_state, action_generator);

            // hashを適当に全て0としているため、重複除去は行わない
            let deduplicator = NoOpDeduplicator;
            let beam_width = FixedBeamWidthSuggester::new(10);

            let (actions, score) = beam.run(input.n, beam_width, deduplicator);

            eprintln!("score: {}", score);
            eprintln!("actions: {:?}", actions);
            assert_eq!(score, 10);
            assert!(actions == vec![1, 3, 2, 0] || actions == vec![2, 3, 1, 0]);
        }
    }
}

mod bayesian {
    use rand::Rng;
    use rand_distr::{Distribution, Gamma, Normal};

    /// 正規-ガンマ分布を表す構造体。
    ///
    /// 正規-逆ガンマ分布  NG(mu, lambda, alpha, beta) = N(mu, (lambda * precision)^-1) * G(alpha, beta) を表す構造体。
    /// ベイズ推定により、正規分布の平均と精度の事後分布を更新することができる。
    #[derive(Debug, Clone, Copy)]
    pub struct GaussInverseGamma {
        mu: f64,
        lambda: f64,
        alpha: f64,
        beta: f64,
    }

    impl GaussInverseGamma {
        /// 正規-ガンマ分布 NG(mu, lambda, alpha, beta) を生成する。
        ///
        /// # Arguments
        ///
        /// * `mu` - 正規分布の平均の事前分布の平均
        /// * `lambda` - サンプリングされた精度と正規分布の平均の精度との比
        /// * `alpha` - **精度 (分散の逆数)** を表すガンマ分布の形状パラメータ
        /// * `beta` - **精度 (分散の逆数)** を表すガンマ分布の尺度パラメータ
        ///
        /// # Note
        ///
        /// * `lambda` は正規分布の平均の事前分布の疑似観測回数と解釈することができる。
        /// * `beta` は正規分布の精度の事前分布の疑似観測回数の2倍と解釈することができる。
        pub fn new(mu: f64, lambda: f64, alpha: f64, beta: f64) -> Self {
            assert!(!mu.is_nan(), "mu is NaN");
            assert!(lambda > 0.0, "lambda is not positive");
            assert!(alpha > 0.0, "alpha is not positive");
            assert!(beta > 0.0, "beta is not positive");

            Self {
                mu,
                lambda,
                alpha,
                beta,
            }
        }

        /// 対象とする正規分布からの疑似観測値から正規-ガンマ分布 NG(mu, lambda, alpha, beta) を生成する。
        ///
        /// # Arguments
        ///
        /// - `mean` - 疑似観測値の期待値
        /// - `std_dev` - 疑似観測値の標準偏差
        /// - `pseudo_observation_count` - 疑似観測回数
        pub fn from_psuedo_observation(
            mean: f64,
            std_dev: f64,
            pseudo_observation_count: usize,
        ) -> Self {
            assert!(std_dev > 0.0, "expected_std_dev is not positive");
            assert!(
                pseudo_observation_count > 0,
                "pseudo_observation_count is not positive"
            );

            let expected_variance = std_dev * std_dev;
            let expected_precision = 1.0 / expected_variance;

            let mu = mean;
            let lambda = pseudo_observation_count as f64;
            let alpha = (pseudo_observation_count * 2) as f64;

            // 精度の期待値E[p] = alpha / beta より、 beta = alpha / E[p]
            let beta = alpha / expected_precision;

            Self::new(mu, lambda, alpha, beta)
        }

        /// 観測値xを元にしてベイズ更新を行う。
        pub fn update(&mut self, x: f64) {
            let mu = (x + self.lambda * self.mu) / (self.lambda + 1.0);
            let lambda = self.lambda + 1.0;
            let alpha = self.alpha + 0.5;
            let dev2 = (x - self.mu) * (x - self.mu);
            let beta = self.beta + 0.5 * (self.lambda * dev2) / (self.lambda + 1.0);

            self.mu = mu;
            self.lambda = lambda;
            self.alpha = alpha;
            self.beta = beta;
        }

        /// (平均, 標準偏差) の期待値を取得する。
        pub fn expected(&self) -> (f64, f64) {
            let expected_precision = self.alpha / self.beta;
            let expected_variance = 1.0 / expected_precision;
            let expected_std_dev = expected_variance.sqrt();
            (self.mu, expected_std_dev)
        }

        pub fn get_pseudo_observation_count(&self) -> f64 {
            self.lambda
        }

        pub fn set_pseudo_observation_count(&mut self, pseudo_observation_count: f64) {
            self.lambda = pseudo_observation_count;
            self.alpha = pseudo_observation_count * 0.5;
        }
    }

    impl Distribution<(f64, f64)> for GaussInverseGamma {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (f64, f64) {
            // ガンマ分布から精度をサンプリング
            let precision = rng.sample(Gamma::new(self.alpha, 1.0 / self.beta).unwrap());
            let std_dev = 1.0 / precision.sqrt();

            // 正規分布から平均をサンプリング
            let std_dev_mean = 1.0 / (precision * self.lambda).sqrt();
            let mean = rng.sample(Normal::new(self.mu, std_dev_mean).unwrap());

            (mean, std_dev)
        }
    }
}
