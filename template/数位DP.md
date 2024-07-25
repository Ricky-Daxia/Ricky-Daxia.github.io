从例题来看：

>   如果一个正整数每一个数位都是 **互不相同** 的，我们称它是 **特殊整数** 。
>
>   给你一个 **正** 整数 `n` ，请你返回区间 `[1, n]` 之间特殊整数的数目。

$f(i,mask,limited,has\_num)$ 

返回从 $i$ 开始填数字， $i$ 前面填的数字集合为 $mask$ ，能构造出的特殊整数数目
$limited$ 表示前面填的数字是否都是 $n$ 对应位上的，如果为 $true$ ，当前位至多为 $int(s[i])$ ，否则至多为 9

$has\_num$ 表示前面是否填了数字（是否跳过），如果为 $true$ ，当前位可以从 0 开始，否则可以跳过，或从 1 开始填数字

入口：$f(0,0,true,false)$ ，如果 $limited$ 为 $false$ ，后面可以任意填，不合理；当前一个数也没填， $has\_num$ 自然是 $false$ 

```c++
class Solution {
public:
//数位DP c++板 dp数组可优化
    int arr[10][1024][2][2]; //dp数组
    string s;
    int dp(int i, int mask, bool limited, bool has_num) {
        if (i == s.size()) return has_num ? 1 : 0;
        if (arr[i][mask][limited][has_num] != 0)
            return arr[i][mask][limited][has_num];
        
        int res = 0;
        if (!has_num) // 选择跳过 不填数字
            res = dp(i + 1, mask, false, false);
        
        int up = limited ? s[i] - '0' : 9; //当位是否受限

        for (int j = 1 - int(has_num); j <= up; j++) // 枚举要填的数字
            if ((mask >> j & 1) == 0) //这一位没有使用过
                res += dp(i + 1, mask | 1 << j, limited && j == up, true);
        
        arr[i][mask][limited][has_num] = res; //记忆化
        return res;
    }
    int countSpecialNumbers(int n) {
        s = to_string(n);
        return dp(0, 0, true, false);
    }
};
// 优化数组 dp[i][mask]
// 有重复运算才记忆化，当 limited 为 true 时，是不会遇到重复子问题的
// 同理，当 has_num 为 false 时，递归过程中只会遇到一次
class Solution {
public:
//数位DP c++板 dp数组可优化
    int arr[10][1024]; //dp数组
    string s;
    int dp(int i, int mask, bool limited, bool has_num) {
        if (i == s.size()) return has_num ? 1 : 0;
        if (!limited && has_num && arr[i][mask] != 0)
            return arr[i][mask];
        
        int res = 0;
        if (!has_num) // 选择跳过 不填数字
            res = dp(i + 1, mask, false, false);
        
        int up = limited ? s[i] - '0' : 9; //当位是否受限

        for (int j = 1 - int(has_num); j <= up; j++) // 枚举要填的数字
            if ((mask >> j & 1) == 0) //这一位没有使用过
                res += dp(i + 1, mask | 1 << j, limited && j == up, true);
        if (!limited && has_num)
            arr[i][mask] = res; //记忆化
        return res;
    }
    int countSpecialNumbers(int n) {
        s = to_string(n);
        return dp(0, 0, true, false);
    }
};
```

---

同样的模板代入这题：

>   给定一个按 **非递减顺序** 排列的数字数组 `digits` 。你可以用任意次数 `digits[i]` 来写的数字。例如，如果 `digits = ['1','3','5']`，我们可以写数字，如 `'13'`, `'551'`, 和 `'1351315'`。
>
>   返回 *可以生成的小于或等于给定整数 `n` 的正整数的个数* 。

区别在于可选的数字在 `digits` 中

```c++
class Solution {
public:
    int atMostNGivenDigitSet(vector<string> &digits, int n) {
        auto s = to_string(n);
        int m = s.length(), dp[m];
        memset(dp, -1, sizeof(dp)); // dp[i] = -1 表示 i 这个状态还没被计算出来
        function<int(int, bool, bool)> f = [&](int i, bool is_limit, bool is_num) -> int {
            if (i == m) return is_num; // 如果填了数字，则为 1 种合法方案
            if (!is_limit && is_num && dp[i] >= 0) return dp[i]; // 在不受到任何约束的情况下，返回记录的结果，避免重复运算
            int res = 0;
            if (!is_num) // 前面不填数字，那么可以跳过当前数位，也不填数字
                // is_limit 改为 false，因为没有填数字，位数都比 n 要短，自然不会受到 n 的约束
                // is_num 仍然为 false，因为没有填任何数字
                res = f(i + 1, false, false);
            char up = is_limit ? s[i] : '9'; // 根据是否受到约束，决定可以填的数字的上限
            // 注意：对于一般的题目而言，如果这里 is_num 为 false，则必须从 1 开始枚举，由于本题 digits 没有 0，所以无需处理这种情况
            for (auto &d : digits) { // 枚举要填入的数字 d
                if (d[0] > up) break; // d 超过上限，由于 digits 是有序的，后面的 d 都会超过上限，故退出循环
                // is_limit：如果当前受到 n 的约束，且填的数字等于上限，那么后面仍然会受到 n 的约束
                // is_num 为 true，因为填了数字
                res += f(i + 1, is_limit && d[0] == up, true);
            }
            if (!is_limit && is_num) dp[i] = res; // 在不受到任何约束的情况下，记录结果
            return res;
        };
        return f(0, true, false);
    }
};
```

---

再看一道题如何转化到这个模板上：

>   给你两个数字字符串 `num1` 和 `num2` ，以及两个整数 `max_sum` 和 `min_sum` 。如果一个整数 `x` 满足以下条件，我们称它是一个好整数：
>
>   -   `num1 <= x <= num2`
>   -   `min_sum <= digit_sum(x) <= max_sum`.
>
>   请你返回好整数的数目。答案可能很大，请返回答案对 `109 + 7` 取余后的结果。
>
>   注意，`digit_sum(x)` 表示 `x` 各位数字之和。

分析：求出不超过 `num2` 中符合条件的数目和，及不超过 `num1` 的数目，做差，再单独考虑 `x==num1` 的情况即可

```c++
class Solution {
public:
    const int MOD = 1e9 + 7;
    int dfs(string s, int min_sum, int max_sum)
    {
        // 只需要两个维度 是因为为 True 的状态只会出现一次
        int n = s.size(), f[n][min(9 * n, max_sum) + 1];
        memset(f, -1, sizeof f);
        function<int(int, int, bool)> g = [&](int i, int sum, bool is_limit)->int
        {
            if (sum > max_sum) return 0;
            if (i == n) return sum >= min_sum;
            if (!is_limit && f[i][sum] != -1) return f[i][sum];
            int res = 0;
            int up = is_limit ? s[i] - '0' : 9;
            for (int d = 0; d <= up; d++)
                res = (res + g(i + 1, sum + d, is_limit && d == up)) % MOD;
            if (!is_limit) f[i][sum] = res;
            return res; 
        };
        return g(0, 0, true);
    }
    int count(string num1, string num2, int min_sum, int max_sum) {
        int res = dfs(num2, min_sum, max_sum) - dfs(num1, min_sum, max_sum) + MOD;
        int sum = 0;
        for (char c: num1) sum += c - '0';
        res += min_sum <= sum && sum <= max_sum;
        return res % MOD;
    }
};
```

---

看一些应用：

>   给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

本题前导零对答案无影响，故可以去掉 $has\_num$ ，把 $mask$ 改成 $cnt$ 表示填了多少个 1

```c++
class Solution {
public:
    int dp[10][10];
    string s;
    int f(int i, int cnt, bool limited)
    {
        if (i == s.size()) return cnt;

        if (!limited && dp[i][cnt]) return dp[i][cnt];

        int res = 0;    

        int up = limited ? s[i] - '0' : 9;
        for (int j = 0; j <= up; j++)
            res += f(i + 1, cnt + (j == 1), limited && j == up);
        
        if (!limited) dp[i][cnt] = res;
        return res;
    }
    int countDigitOne(int n) {
        s = to_string(n);
        return f(0, 0, true);
    }
};
```

---

例二：

>   给定一个正整数 `n` ，请你统计在 `[0, n]` 范围的非负整数中，有多少个整数的二进制表示中不存在 **连续的 1** 。

本题难点在于从二进制的最左向最右枚举

```c++
class Solution {
public:
    int dp[32][2];
    int m, n;
    int f(int i, bool pre, bool limited)
    {
        if (i < 0) return 1;
        if (!limited && dp[i][pre]) return dp[i][pre];
        int up = limited ? n >> i & 1 : 1;
        int res = f(i - 1, false, limited && up == 0); // 0
        if (!pre && up == 1) res += f(i - 1, true, limited); // 1
        if (!limited) dp[i][pre] = res;
        return res;
    }
    int findIntegers(int n) {
        this->n = n;
        m = __lg(n);
        return f(m, false, true);
    }
};
```

---

```c++
// 统计范围内的步进数字数目
class Solution {
public:
    int calc(string& s)
    {
        int n = s.size();
        LL f[n][10];
        memset(f, -1, sizeof f);
        function<int(int, int, bool, bool)> dfs = [&](int i, int pre, bool limited, bool is_num)
        {
            if (i == n) return is_num ? 1LL : 0LL;
            if (!limited && is_num && f[i][pre] != -1) return f[i][pre];
            LL res = 0;
            if (!is_num) res = dfs(i + 1, pre, false, false);
            int up = limited ? s[i] - '0' : 9;
            for (int d = 1 - is_num; d <= up; d++)
                if (!is_num || abs(d - pre) == 1)
                    res = (res + dfs(i + 1, d, limited && d == up, true)) % MOD;
            if (!limited && is_num) f[i][pre] = res;
            return res;
        };
        return dfs(0, 0, true, false);
    }
    int countSteppingNumbers(string low, string high) {
        auto valid = [](string& s)
        {
            for (int i = 1; i < s.size(); i++)
                if (abs(s[i] - '0' - (s[i - 1] - '0')) != 1) return false;
            return true;
        };
        return (calc(high) - calc(low) + MOD + (int)valid(low)) % MOD;
    }
};
```

---

![image-20230915211608371](https://cdn.jsdelivr.net/gh/Ricky-Daxia/Hei_Xiu/202407252335819.png)

```c++
int dfs(int i, int mx, bool is_num, bool limited)
{
    if (i == n) return mx; // 这里统计的是 mx 值之和 所以返回 mx
    if (!limited && is_num && f[i][mx] != -1) return f[i][mx];
    int res = 0;
    if (!is_num) res += dfs(i + 1, mx, false, false);
    int up = limited ? s[i] - '0' : 9;
    for (int d = 1 - is_num; d <= up; d++)
        res = (res + dfs(i + 1, max(d, mx), true, limited && d == up)) % MOD;
    if (!limited && is_num) f[i][mx] = res;
    return res;
}
```

---

![image-20231025184827489](https://cdn.jsdelivr.net/gh/Ricky-Daxia/Hei_Xiu/202407252336554.png)

自己做出来了，细节是最后要加模再取模

```c++
string s;
int n;
int f[210][10][2010];

int dfs(int i, int sum, int last, bool is_num, bool limited)
{
    if (i == n) return is_num && last != 0 && sum % last == 0;
    if (!limited && is_num && f[i][last][sum] != -1) return f[i][last][sum];
    int res = 0;
    if (!is_num) res += dfs(i + 1, 0, 0, false, false);
    int up = limited ? s[i] - '0' : 9;
    for (int d = 1 - is_num; d <= up; d++)
        res = ((LL)res + dfs(i + 1, sum + d, d, true, limited && d == up)) % mod;
    if (!limited && is_num) f[i][last][sum] = res;
    return res;
}

int main()
{
    cin >> s;
    n = s.size();
    memset(f, -1, sizeof f);
    int a = dfs(0, 0, 0, false, true);
    int sum = 0;
    for (int i = 0; i < n - 1; i++) 
        sum += s[i] - '0';
    if (s[n - 1] != '0' && sum % (s[n - 1] - '0') == 0) a --;
    cin >> s;
    n = s.size();
    memset(f, -1, sizeof f);
    int b = dfs(0, 0, 0, false, true);
    cout << ((LL)b - a + mod) % mod << endl;
    return 0;
}
```

---

### 数位 DP 2.0 模板

```python
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:

"""     
        # 基础版 只有两个参数 只支持上界

        high = str(finish) # 转字符串
        n = len(high)
        
        @cache
        def dfs(i: int, limit_high: bool) -> int: # 当前填到哪一位 当前位有无限制
            if i == n:
                return 1
                
            # 第 i 个数位可以从哪枚举到哪
            lo = 0
            hi = int(high[i]) if limit_high else 9
            
            res = 0
            for d in range(lo, hi + 1):
                res += dfs(i + 1, limit_high and d == hi)
            return res
        
        dfs(0, True)
"""

"""     
        # 基础版 支持上下界
        
        low = str(start)
        high = str(finish) # 转字符串
        n = len(high)
        low = '0' * (n - len(low)) + low # 补前导零
        
        @cache
        def dfs(i: int, limit_low: bool, limit_high: bool) -> int: # 当前填到哪一位 当前位有无限制
            if i == n:
                return 1
                
            # 第 i 个数位可以从哪枚举到哪
            lo = int(low[i]) if limit_high else 0
            hi = int(high[i]) if limit_high else 9
            
            res = 0
            for d in range(lo, hi + 1):
                res += dfs(i + 1, limit_low and d == lo, limit_high and d == hi)
            return res
        
        dfs(0, True, True)
"""

"""
        ### 本题代码

        low = str(start)
        high = str(finish) # 转字符串
        n = len(high)
        low = '0' * (n - len(low)) + low # 补前导零
        diff = n - len(s)
        
        @cache
        def dfs(i: int, limit_low: bool, limit_high: bool) -> int: # 当前填到哪一位 当前位有无限制
            if i == n:
                return 1
                
            # 第 i 个数位可以从哪枚举到哪
            # 如果对数位有其他约束 应当只在下面的 for 循环做限制
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9 
            
            res = 0
            if i < diff:
                for d in range(lo, min(hi, limit) + 1):
                    res += dfs(i + 1, limit_low and d == lo, limit_high and d == hi)
            else:
                # 必须填 s[i - diff]
                x = int(s[i - diff])
                if lo <= x <= min(hi, limit):
                    res = dfs(i + 1, limit_low and x == lo, limit_high and x == hi)
            return res
        
        return dfs(0, True, True)
"""

        low = str(start)
        high = str(finish) # 转字符串
        n = len(high)
        low = '0' * (n - len(low)) + low # 补前导零
        diff = n - len(s)
        
        # 支持前导 0 怎么改 加上 is_num
        # is_num: 前面是否填了非零数字
        @cache
        def dfs(i: int, limit_low: bool, limit_high: bool, is_num: bool) -> int: # 当前填到哪一位 当前位有无限制
            if i == n:
                return 1 if is_num else 0
            
            res = 0
            if not is_num and low[i] == '0': # 前面填的都是 0 limit_low 一定是 True
                # 这一位可以为 0
                if i < diff:
                	res = dfs(i + 1, True, False, False)
                
            # 第 i 个数位可以从哪枚举到哪
            # 如果对数位有其他约束 应当只在下面的 for 循环做限制
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9 
            
            d0 = 0 if is_num else 1
            if i < diff:
                for d in range(max(lo, d0), min(hi, limit) + 1):
                    res += dfs(i + 1, limit_low and d == lo, limit_high and d == hi, True)
            else:
                # 必须填 s[i - diff]
                x = int(s[i - diff])
                if max(d0, lo) <= x <= min(hi, limit):
                    res = dfs(i + 1, limit_low and x == lo, limit_high and x == hi, True)
            return res
        
        return dfs(0, True, True, False)
```

C++ 模板

```c++
class Solution {
public:
    long long numberOfPowerfulInt(long long start, long long finish, int limit, string s) {
        string low = to_string(start);
        string high = to_string(finish);
        int n = high.size();
        low = string(n - low.size(), '0') + low; // 补前导零，和 high 对齐
        int diff = n - s.size();

        vector<long long> memo(n, -1);
        function<long long(int, bool, bool)> dfs = [&](int i, bool limit_low, bool limit_high) -> long long {
            if (i == low.size()) {
                return 1;
            }

            if (!limit_low && !limit_high && memo[i] != -1) {
                return memo[i]; // 之前计算过
            }

            // 第 i 个数位可以从 lo 枚举到 hi
            // 如果对数位还有其它约束，应当只在下面的 for 循环做限制，不应修改 lo 或 hi
            int lo = limit_low ? low[i] - '0' : 0;
            int hi = limit_high ? high[i] - '0' : 9;

            long long res = 0;
            if (i < diff) { // 枚举这个数位填什么
                for (int d = lo; d <= min(hi, limit); d++) {
                    res += dfs(i + 1, limit_low && d == lo, limit_high && d == hi);
                }
            } else { // 这个数位只能填 s[i-diff]
                int x = s[i - diff] - '0';
                if (lo <= x && x <= min(hi, limit)) {
                    res = dfs(i + 1, limit_low && x == lo, limit_high && x == hi);
                }
            }

            if (!limit_low && !limit_high) {
                memo[i] = res; // 记忆化 (i,false,false)
            }
            return res;
        };
        return dfs(0, true, true);
    }
};

```

---

