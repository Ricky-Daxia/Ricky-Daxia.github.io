logTrick 主要是将维护一系列左端点转化为维护一系列区间（固定右端点时，给定运算下对应一段连续的左端点），适用于运算具有单调性，且区间数不是很多的情况（例如 gcd 最多 $logU$ 种）

### 模板
以 [LC3171](https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/description/) 为例题

-   求出**所有**子数组的按位或的结果，以及值等于该结果的子数组的个数
-   求按位或结果等于**任意给定数字**的子数组的最短长度/最长长度

可用于 `and`、`or`、`gcd`、`lcm`

特点：固定右端点，结果具有单调性质，相同结果的左端点会形成一段连续区间

```cpp
        map<int, LL> res; // 统计子数组数量
        vector<array<int, 3>> a; // 左端点闭区间 [a[0], a[1]] 值为 a[2]
        for (int i = 0; i < nums.size(); i++) {
            int cur = nums[i];
            for (auto &v: a) {
                v[2] |= cur; // 给定运算
            }
            a.push_back({i, i, cur});
            int p = 0; // 原地去重
            for (int i = 1; i < a.size(); i++) {
                if (a[p][2] != a[i][2]) {
                    p ++;
                    a[p] = a[i];
                } else {
                    a[p][1] = a[i][1];
                }
            }
            a.resize(p + 1);
            // 这里求的是大于等于 k 的最短子数组长度
            for (auto &t: a) {
                if (t[2] >= k) {
                    ans = min(ans, i - t[1] + 1); 
                }
            }
            // 累加子数组
            for (auto &t: a) {
                res[t[2]] += t[1] - t[0] + 1;
            }
        }
```

另一个模板是枚举 $logX$ 种 `and`、`or` 值，找到每个二进制位不同的最近位置

以 [LC3209](https://leetcode.cn/problems/number-of-subarrays-with-and-value-of-k/description/) 为例，求出按位与为 $k$ 的子数组数量

```cpp
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int K) {
        int n = nums.size();
        const int MAXP = 30;

        // last[i][p]：在位置 i 的左边（含位置 i），二进制第 p 位是 0 的最近位置在哪
        int last[n][MAXP];
        for (int j = 0; j < MAXP; j++) {
            if (nums[0] >> j & 1) last[0][j] = -1;
            else last[0][j] = 0;
        }
        for (int i = 1; i < n; i++) for (int j = 0; j < MAXP; j++) {
            last[i][j] = last[i - 1][j];
            if (nums[i] >> j & 1 ^ 1) last[i][j] = i;
        }

        long long ans = 0;
        // 枚举子数组右端点
        for (int i = 0; i < n; i++) {
            // 对于二进制的每一位，拿出上一个 0 在什么位置
            vector<int> pos;
            for (int j = 0; j < MAXP; j++) if (last[i][j] >= 0) pos.push_back(last[i][j]);
            sort(pos.begin(), pos.end());
            pos.resize(unique(pos.begin(), pos.end()) - pos.begin());
            reverse(pos.begin(), pos.end());
            // 枚举 logX 种 AND 值
            int v = nums[i];
            for (int j = 0; j < pos.size(); j++) {
                v &= nums[pos[j]];
                if (v < K) break;
                if (v == K) {
                    // 发现了目标值，求一下这一段子数组的长度
                    if (j + 1 == pos.size()) ans += pos[j] + 1;
                    else ans += pos[j] - pos[j + 1];
                }
            }
        }
        return ans;
    }
};
```

因为具有单调性，还可以考虑滑窗来做，但是运算不具有可逆性的时候，需要用到一个栈来维护信息

讲解见 [LC3171这篇题解](https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/solutions/2798206/li-yong-and-de-xing-zhi-pythonjavacgo-by-gg4d) 

```cpp
class Solution {
public:
    int minimumDifference(vector<int>& nums, int k) {
        int ans = INT_MAX, left = 0, bottom = 0, right_or = 0;
        for (int right = 0; right < nums.size(); right++) {
            right_or |= nums[right];
            while (left <= right && (nums[left] | right_or) > k) {
                ans = min(ans, (nums[left] | right_or) - k);
                left++;
                if (bottom < left) {
                    // 重新构建一个栈
                    for (int i = right - 1; i >= left; i--) {
                        nums[i] |= nums[i + 1];
                    }
                    bottom = right;
                    right_or = 0;
                }
            }
            if (left <= right) {
                ans = min(ans, k - (nums[left] | right_or));
            }
        }
        return ans;
    }
};
```

