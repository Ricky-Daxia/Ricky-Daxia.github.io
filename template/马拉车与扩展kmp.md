### Manacher

用途：求以 i 为中心的回文子串的长度

技巧：加分隔符隔开每个字符

核心思路

-   维护一个 box，记录当前已知的最靠右的回文子串的左右端点 [l,r]
-   计算 d[i] 时，如果 i 在 box 内，就要找对称位置，l+(r-i)
    -   如果对称点的回文半径不超出 box，放心转移 d[i]=d[l+r-i]
    -   如果超出，就保守估计，d[i]=r-i+1
    -   看视频的图理解
-   在 box 外就暴力枚举
-   求出 d[i] 后，看 box 的边界是否需要更新

来自 oi-wiki 的代码

```c++
// 先预处理成 #a#b#c#
vector<int> d(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
  int k = (i > r) ? 1 : min(d[l + r - i], r - i + 1);
  while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
    k++;
  }
  d[i] = k--;
  if (i + k > r) {
    l = i - k;
    r = i + k;
  }
}
```

代码中 d[i] 表示 s（原串） 中以 i 为中心的极大回文子串的 **总长度+1**

如果要求 s 的最长回文子串长度怎么办？

```c++
for (int i = 0; i < n; i++) {
    mx = max(mx, d[i] - 1);
}
```

**任意插入最少字符，使得串变回文：**

**答案是 n - 最长回文子串长度**

---

### 扩展 kmp

Z 函数：对于字符串 s，z[i] 表示 s 与其后缀 s[i,n] 的最长公共前缀的长度

核心思路：

-   定义匹配段为区间 [i,i+z[i]-1]，维护一个 box 表示最靠右的匹配段的左右端点 [l,r]，那么 [l,r] 是 s 的前缀
-   计算 z[i] 时，同样分情况讨论，情况同马拉车中的讨论
    -   在 box 内且不超出，z[i]=z[i-l+1]
    -   超出时则令为 z[i]=r-i+1
    -   其余情况暴力枚举

来自 oi-wiki 的代码

```c++
vector<int> z_function(string s) {
  int n = (int)s.length();
  vector<int> z(n);
  for (int i = 1, l = 0, r = 0; i < n; ++i) {
    if (i <= r && z[i - l] < r - i + 1) {
      z[i] = z[i - l];
    } else {
      z[i] = max(0, r - i + 1);
      while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
    }
    if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
  }
  return z;
}
```

代码中规定 z[0] = 0，但实际上 z[0]=n，在使用模板时需注意

### Z 函数的应用

给定串 $s$ 和 $p$，如何求出 $s[l:r]$ 与 $p$ 的最长公共前缀和最长公共后缀呢？

设 $p$ 的长度为 $m$

对于前缀，可以通过获取 $p+s$ 的 $z$ 数组来获得，有 $pre_l=z[m+l]$

对于后缀，可以通过获取 $reverse(p) + reverse(s)$ 的 $z$ 数组并反转来获得，有 $suf_r=z[r]$

例题：对于 $s$ 的长为 $m$ 的窗口，能否做到连续修改 $k$ 个字符，使得 $s[l:r]=p$？

```c++
class Solution {
    vector<int> calc_z(string s) {
        int n = s.length();
        vector<int> z(n);
        int box_l = 0, box_r = 0; // z-box 左右边界
        for (int i = 1; i < n; i++) {
            if (i <= box_r) {
                z[i] = min(z[i - box_l], box_r - i + 1);
            }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                box_l = i;
                box_r = i + z[i];
                z[i]++;
            }
        }
        return z;
    }

public:
    int minStartingIndex(string s, string pattern) {
        vector<int> pre_z = calc_z(pattern + s);
        ranges::reverse(pattern);
        ranges::reverse(s);
        vector<int> suf_z = calc_z(pattern + s);
        ranges::reverse(suf_z); // 也可以不反转，下面写 suf_z[suf_z.size() - i]
        int m = pattern.length();
        for (int i = m; i <= s.length(); i++) {
            if (pre_z[i] + suf_z[i - 1] >= m - k) {
                return i - m;
            }
        }
        return -1;
    }
};
```