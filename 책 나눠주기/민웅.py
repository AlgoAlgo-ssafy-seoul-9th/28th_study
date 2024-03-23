# 9576_책나눠주기_Handind out books
import sys
import heapq
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    hq = []
    for _ in range(M):
        s, g = map(int, input().split())
        heapq.heappush(hq, [g, s])
    books = [0]*(N+1)
    ans = 0
    while hq:
        a, b = heapq.heappop(hq)
        for i in range(b, a+1):
            if not books[i]:
                ans += 1
                books[i] = 1
                break
    print(ans)