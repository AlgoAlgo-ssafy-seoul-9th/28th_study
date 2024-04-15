import sys
input = sys.stdin.readline


N = int(input())
nums = [0] * (N+1)

for i in range(N):
    a, b = map(int, input().split())
    nums[i] = a
    nums[i+1] = b

dp = [[0] * (N+1) for _ in range(N+1)]

for cnt in range(1, N+1):
    for i in range(1, N+1-cnt):
        j = i + cnt
        minv = 500**3
        for p in range(i, j):
            minv = min(minv, dp[i][p] + dp[p+1][j] + nums[i-1]* nums[p] * nums[j])
        
        dp[i][j] = minv

print(dp[1][N])