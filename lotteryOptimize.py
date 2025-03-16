from typing import List, Tuple    
import itertools

class LotteryOptimizer:
    def __init__(self):
        self.covering_design_vals = [
            1,3,3,3,4,6,6,7,7,10,10,12,12,15,16,17,19,21,22,23,24,27,28,30,31,
            31,38,39,40,42,47,50,51,54,55,59,63,65,67,70,73,79,80,82,87,90,96,
            98,99,105,110,114,117,119,128,132,135,140,142,143,143,157,160,163,172
        ]

    def compute_lottery_numbers_in_range(self, n_min: int, n_max: int) -> None:
        """Compute L(n,6,6,2) for n_min <= n <= n_max."""
        prev_lotto_num = 1
        n = n_min
        
        while n <= n_max:
            new_lotto_num = self.possible_lottery_number(n, prev_lotto_num)
            if n == n_max:
                break
            prev_lotto_num = new_lotto_num
            n += 1

    def possible_lottery_number(self, n: int, prev_lotto_num: int) -> int:
        """Calculate possible lottery number for given n."""
        ub = self.calculate_upper_bound(n, prev_lotto_num)
        
        if prev_lotto_num == ub:
            print(f"L({n},6,6,2) = {ub}")
            return ub
            
        rs = self.bound_isolated_blocks(n, ub)
        min_num_i_blocks = self.get_min_num_iblocks(n)
        
        slim_i_exceptions = []
        delta_exceptions = []
        
        for r in rs:
            # Get slim I exceptions
            exception = self.get_slim_i_exceptions(n, ub, r)
            if self.interval_check(exception):
                slim_i_exceptions.append(exception)
            
            # Get Delta(I) exceptions
            delta_exc = self.get_deltaI_exceptions(n, ub, min_num_i_blocks, r)
            delta_exceptions.extend(delta_exc)
        
        self.write_lotto_result(slim_i_exceptions, delta_exceptions, n, ub)
        return ub

    def calculate_upper_bound(self, n: int, guess: int) -> int:
        """Calculate upper bound using covering design approach."""
        while True:
            solutions = []
            for vs in self.generate_valid_combinations(n, guess):
                if self.test_upper_bound(vs, n, guess):
                    solutions.append(vs)
            
            if solutions:
                return guess
            guess += 1

    def test_upper_bound(self, vs: List[int], n: int, ub: int) -> bool:
        """Test if a combination satisfies upper bound conditions."""
        if len(vs) != 5:
            return False
        
        if not all(1 <= v <= 65 for v in vs):
            return False
            
        m = n - 25
        if sum(vs) != m:
            return False
            
        scores = [self.covering_design_vals[v-1] for v in vs]
        if sum(scores) > ub:
            return False
            
        return vs == sorted(vs)

    def get_min_num_iblocks(self, n: int) -> int:
        """Calculate minimum number of I-blocks."""
        if (n - 5) % 5 == 0:
            return (n - 5) // 5
        return 1 + ((n - 5) // 5)

    def bound_isolated_blocks(self, n: int, ub: int) -> List[int]:
        """Calculate bounds for isolated blocks."""
        d1_min = self.get_r_min(n, ub)
        return [r for r in range(5) if self.bound_isolated_blocks_helper(n, ub, d1_min, r)]

    def get_r_min(self, n: int, ub: int) -> int:
        """Calculate r_min value."""
        t = 2 * n - 6 * (ub - 1)
        return 0 if t <= 0 else t

    def bound_isolated_blocks_helper(self, n: int, ub: int, d1_min: int, r: int) -> bool:
        """Helper function for bound_isolated_blocks."""
        if r == 0:
            return d1_min < 1
        
        if d1_min > 6 * r:
            return False
            
        p = 6 - r
        lb = ub - r
        m = n - 6 * r
        
        return self.check_furedi_lower_bound(m, 6, p, lb)

    def check_furedi_lower_bound(self, n: int, k: int, p: int, lb: int) -> bool:
        """Check Füredi lower bound conditions."""
        llb = k * lb
        np = p - 1
        
        for combination in itertools.combinations_with_replacement(range(1, n + 1), np):
            if sum(combination) == n:
                ws = []
                for x in combination:
                    if (x - 1) % (k - 1) == 0:
                        ws.append(x * ((x - 1) // (k - 1)))
                    else:
                        ws.append(x * (1 + ((x - 1) // (k - 1))))
                if sum(ws) <= llb:
                    return True
        return False

    def get_slim_i_exceptions(self, n: int, ub: int, r: int) -> Tuple[int, List[int]]:
        """Calculate slim I exceptions."""
        d2_upper = 9 * (4 - r)
        d2_lower = max(0, (9 * (4 * n - 6 * (ub - 1) - 18 * r - 16 * (4 - r))) // 4)
        return [r, [d2_lower, d2_upper]]

    def interval_check(self, exception: List) -> bool:
        """Check if interval is valid."""
        return exception[1][0] <= exception[1][1]

    def get_deltaI_exceptions(self, n: int, ub: int, min_num_i_blocks: int, r: int) -> List[List[int]]:
        """Calculate Delta(I) exceptions."""
        delta2 = self.get_delta_two(n, ub, r)
        
        # Generate base deltas with 1s and 2s
        deltas = []
        for ones in range(6):
            for twos in range(6 - ones):
                if ones + twos > 5:
                    continue
                    
                delta = [1] * ones + [2] * twos
                threes = 5 - (ones + twos)
                if threes > 0:
                    delta.extend([3] * threes)
                
                if sum(delta) >= min_num_i_blocks:
                    base_excess = self.get_base_excess(n, r, ub)
                    if self.can_populate_toes(n, r, base_excess, delta):
                        deltas.append(delta)
        
        return deltas

    def get_delta_two(self, n: int, ub: int, r: int) -> int:
        """Calculate delta two value."""
        x = 3 * n - 12 * r - 6 * (ub - 1)
        if x % 9 == 0:
            return max(0, x // 9)
        return max(0, 1 + (x // 9))

    def get_base_excess(self, n: int, r: int, ub: int) -> int:
        """Calculate base excess."""
        return 6 * (ub - 1) + 6 * r - 2 * n

    def can_populate_toes(self, n: int, r: int, base_excess: int, delta: List[int]) -> bool:
        """Check if toes can be populated in I-blocks."""
        b = sum(delta)
        min_toes = 2 * n - 10 - 5 * (b + r)
        
        num_ones_and_twos = len([x for x in delta if x in (1, 2)])
        num_threes = 5 - num_ones_and_twos
        excess = base_excess - num_threes
        
        delta_no_ones = [x for x in delta if x != 1]
        return self.check_toe_population(delta_no_ones, min_toes, excess)

    def check_toe_population(self, delta_no_ones: List[int], min_toes: int, excess: int) -> bool:
        """Check toe population constraints."""
        vs = [0] * len(delta_no_ones)
        return self.backtrack_toe_population(vs, 0, delta_no_ones, min_toes, excess)

    def backtrack_toe_population(self, vs: List[int], idx: int, delta_no_ones: List[int], 
                               min_toes: int, excess: int) -> bool:
        """Backtracking algorithm for toe population."""
        if idx == len(vs):
            return sum(vs) >= min_toes and self.calculate_total_excess(vs, delta_no_ones) <= excess
            
        prev_val = vs[idx-1] if idx > 0 else 0
        max_val = 10 if delta_no_ones[idx] == 2 else 15
        
        for val in range(prev_val, max_val + 1):
            vs[idx] = val
            if self.backtrack_toe_population(vs, idx + 1, delta_no_ones, min_toes, excess):
                return True
        return False

    def calculate_total_excess(self, vs: List[int], delta_no_ones: List[int]) -> int:
        """Calculate total excess for toe population."""
        excess = 0
        for v, d in zip(vs, delta_no_ones):
            if d == 2:
                ex_vals = [0,0,0,0,0,0,2,3,7,10]
                excess += ex_vals[min(v, len(ex_vals)-1)]
            else:  # d == 3
                ex_vals = [1,2,3,4,5,6,7,8,9,10,11,12,20,25,27]
                excess += ex_vals[min(v, len(ex_vals)-1)]
        return excess

    def write_lotto_result(self, slim_i_exceptions: List, delta_exceptions: List, n: int, ub: int) -> None:
        """Write lottery result with exceptions."""
        if not slim_i_exceptions and not delta_exceptions:
            print(f"L({n},6,6,2) = {ub}")
            return
            
        print(f"We conjecture that L({n},6,6,2) = {ub} and must rule out the following cases")
        
        for r, [d2l, d2u] in slim_i_exceptions:
            d1 = 6 * r
            if d2l == d2u:
                print(f"d_1 = {d1} and d_2 = {d2l}")
            else:
                print(f"d_1 = {d1} and d2 in the range [{d2l},{d2u}]")
                
        for delta in delta_exceptions:
            num_twos = len([x for x in delta if x == 2])
            d2 = num_twos * 9
            print(f"d_2 <= {d2} and δ(I) = {delta}")

if __name__ == "__main__":
    optimizer = LotteryOptimizer()
    optimizer.compute_lottery_numbers_in_range(1, 172)