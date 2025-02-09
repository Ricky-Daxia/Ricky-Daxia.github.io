---
title: 题单记录
date: 2025-02-09
categories:
  - 算法
tags:
  - 题目
plugins:
  - mathjax
description: 积累一些题单中的题
---

[CF1583C](https://codeforces.com/problemset/problem/1538/C)

x<=ai+aj<=y的对数 =  ai+aj<=y的对数 - ai+aj<=x-1的对数 

用双指针求解数对：i 和 j 分别指向最左和最右，移动 i 的时候，j 只会往左移

[LC795](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/description/)

最大值在[a,b]范围内的子数组数目 = 最大值小于等于b的子数组数目 - 最大值小于a的子数组数目

枚举左端点即可

[LC992](https://leetcode.cn/problems/subarrays-with-k-different-integers/description/)

种类恰好为k的子数组数目 = 种类小于等于k-1的子数组数目 - 种类小于等于k的子数组数目

滑窗