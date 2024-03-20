import sys
input = sys.stdin.readline

N, K = map(int, input().split())
menus = {}
ans = -1
for _ in range(N):
    food = int(input())
    if food in menus.keys():
        menus[food] += 1
    else:
        menus[food] = 1
    
    if menus[food] >= K:
        if ans == -1:
            ans = food
        else:
            if food < ans:
                ans = food
    
    print(ans)