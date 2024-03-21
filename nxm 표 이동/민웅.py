import sys
input = sys.stdin.readline
from collections import deque
dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]

N, M = map(int, input().split())

field = [list(map(int, input().split())) for _ in range(N)]
visited = [[[0, 0] for _ in range(M)] for _ in range(N)]

q = deque()
# i, j, cnt, 벽통과
if field[0][0] == 1:
    q.append([0, 0, 1, 1])
    visited[0][0][1] = 1
else:
    q.append([0, 0, 1, 0])
    visited[0][0][0] = 0

ans = float('inf')
while q:
    x, y, cnt, one = q.popleft()
    if x == N-1 and y == M-1:
        if cnt < ans:
            ans = cnt
        break
    
    for d in dxy:
        nx = x + d[0]
        ny = y + d[1]

        if 0 <= nx <= N-1 and 0 <= ny <= M-1:
            if field[nx][ny] == 0:
                if not visited[nx][ny][one]:
                    q.append([nx, ny, cnt+1, one])
                    visited[nx][ny][one] = cnt+1
            else:
                if not one and not visited[nx][ny][1]:
                    q.append([nx, ny, cnt+1, 1])
                    visited[nx][ny][1] = cnt+1

if ans == float('inf'):
    print(-1)
else:
    print(ans)