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

