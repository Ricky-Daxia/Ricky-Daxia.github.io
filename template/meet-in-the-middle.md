#### 买瓜

> 小蓝正在一个瓜摊上买瓜。
>
> 瓜摊上共有 $n$ 个瓜，每个瓜的重量为 $A_i$ 。
>
> 小蓝刀功了得，他可以把任何瓜劈成完全等重的两份，不过每个瓜只能劈一刀。
>
> 小蓝希望买到的瓜的重量的和恰好为 $m$ 。
>
> 请问小蓝至少要劈多少个瓜才能买到重量恰好为 $m$ 的瓜。
>
> 如果无论怎样小蓝都无法得到总重恰好为 $m$ 的瓜，请输出 `−1`。
>
> 对于 20% 的评测用例，$∑n≤10$ ；
> 对于 60% 的评测用例，$∑n≤20$ ；
> 对于所有评测用例，$1≤n≤30$，$1≤A_i≤10^9$，$1≤m≤10^9$ 。

一份能过洛谷的折半搜索代码，自己实现了哈希表（若 `sum` 爆 int ，可考虑将 `sum` 变量换成 `LL` ）

```c++
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 15000010; // 3^(n/2) <= 143489073 
int h[N], e[N], ne[N], idx = 1;
int v[N];
const int MOD = 9998777;

void insert(int key, int val)
{
    int id = key % MOD;
    v[idx] = val;
    e[idx] = key;
    ne[idx] = h[id];
    h[id] = idx ++;
}

int find(int x)
{
    if (x < 0) return -1;
    for (int i = h[x % MOD]; i; i = ne[i])
    {
        if (e[i] == x) return i;
    }
    return -1;
}

int n, m;
int a[35];
int res;

void dfs(int l, int r, int sum, int cnt)
{
    if (sum > m) return;
    if (l > r)
    {
        int t = find(sum);
        if (t == -1) insert(sum, cnt);
        else v[t] = min(v[t], cnt);
    }
    else
    {
        dfs(l + 1, r, sum, cnt);
        dfs(l + 1, r, sum + a[l], cnt);
        dfs(l + 1, r, sum + a[l] / 2, cnt + 1);
    }
}

void dfs2(int l, int r, int sum, int cnt)
{
    if (sum > m || cnt > res) return;
    if (l > r)
    {
        int t = find(m - sum);
        if (t != -1) res = min(res, v[t] + cnt);
        return;
    }
    else
    {
        dfs2(l + 1, r, sum, cnt);
        dfs2(l + 1, r, sum + a[l], cnt);
        dfs2(l + 1, r, sum + a[l] / 2, cnt + 1);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    m *= 2;
    for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &a[i]);
        a[i] *= 2;
    }
    
    res = 0x3f3f3f3f;
    sort(a + 1, a + 1 + n, [](int x, int y)->bool {return x > y;});
    dfs(1, n / 2, 0, 0);
    dfs2(n / 2 + 1, n, 0, 0);
    
    if (res == 0x3f3f3f3f) puts("-1");
    else printf("%d\n", res);
    return 0;
}
```



---

#### [Leetcode956最高的广告牌][https://leetcode.cn/problems/tallest-billboard/description/]

题意是给定一个数组，选两个子集出来作为支架的长度，要保证左右支架长度相等，求最大长度。

数据范围是 $n\leq 20$ ，$sum(rods[i]) \leq 5000$ ，从数据范围中可以得到提示：总和这么小，那么 $sum$ 很可能作为一个需要利用的信息。本题思考的核心是**左右支架高度的差值**。

三种思考的角度：

1. 每个数（1）放左边（2）放右边（3）跳过，令 $a$ 为左边的和， $b$ 为右边的和，那么要求的就是当 $a-b=0$ 时，$a$ 的最大值。可以转换为选（1）就 **+** ，选（2）就 **-** ，否则跳过，然后得出一种用哈希表记录状态的DP算法：

    ```c++
    int tallestBillboard(vector<int>& rods) {
        unordered_map<int, int> f;
        f[0] = 0;
        // 键值 (k, v) 表示 （总和，正数和）
        // 这里 总和 表示 正数和与负数和的差
        for (int x: rods)
        {
            unordered_map<int, int> g(f);
            for (auto& t: g)
            {
                int s = t.first, a = t.second;
                f[s + x] = max(f[s + x], a + x);
                f[s - x] = max(f[s - x], a);
            }
        }
        return f[0];
    ```

2. 从一般的DP角度考虑，可以想到定义 $f[i][j]$ 表示前 $i$ 个数组成的高度差为 $j$ 的时候的长度之和，下面是一个压缩成一维数组的版本，注意体会细节

    ```c++
    int tallestBillboard(vector<int>& rods) {
        int n = rods.size();
        int sum = accumulate(rods.begin(), rods.end(), 0);
        vector<int> f(sum + 1);
    
        for (int i = 1; i <= n; i++)
        {
            auto g = f;
            for (int j = 0; j <= sum; j++)
            {
                // 钢筋高度差为 j 的时候 长度和至少为 j
                if (f[j] < j) continue;
                // 加到长的一侧
                int k = j + rods[i - 1]; // 高者更高
                g[k] = max(g[k], f[j] + rods[i - 1]);
                // 加到短的一侧
                k = abs(j - rods[i - 1]); // 巧妙
                g[k] = max(g[k], f[j] + rods[i - 1]);
            }
            f = g;
        }
        return f[0] / 2;
    }
    ```

3. 复习下折半查找

    ```c++
    int tallestBillboard(vector<int>& rods) {
        int n = rods.size();
        unordered_map<int, int> m1, m2;
        // (delta, score) delta 表示整数与负数的和
        function<void(int, int, int, int, unordered_map<int, int>&)> dfs = 
          [&](int l, int r, int sum, int s, unordered_map<int, int>& m) 
        {
            if (l > r) 
            {
                m[sum] = max(m[sum], s);
                return;
            }
            dfs(l + 1, r, sum, s, m); // 不选
            dfs(l + 1, r, sum + rods[l], s + rods[l], m); // 放左边
            dfs(l + 1, r, sum - rods[l], s, m); // 放右边
        };
    
        dfs(0, n / 2, 0, 0, m1);
        dfs(n / 2 + 1, n - 1, 0, 0, m2);
    
        int res = 0;
        for (auto& [sum, s]: m1)
        {
            if (m2.count(-sum))
                res = max(res, s + m2[-sum]);
        }
        return res;
    }
    ```




---

