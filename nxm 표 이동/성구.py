import sys
from collections import deque
input = sys.stdin.readline

N, M = map(int, input().split())

field = tuple(tuple(map(int, input().split())) for _ in range(N))

def bfs():
    que = deque([(0,0,field[0][0])])  
    visited = [[[0,0] for _ in range(M)] for _ in range(N)]  
    visited[0][0][field[0][0]] = 1
    while que:
        i, j, is_visit = que.popleft()
        if i == N-1 and j == M-1:
            return visited[i][j][is_visit]
        
        for di, dj in [(1,0), (0,1), (-1,0), (0,-1)]:
            ni,nj = i+di, j+dj
            if 0 <= ni < N and 0 <= nj < M:
                if not visited[ni][nj][is_visit]:
                    if is_visit:
                        if not field[ni][nj]:
                            visited[ni][nj][1] = visited[i][j][1] + 1
                            que.append((ni,nj,1))
                    else:
                        visited[ni][nj][field[ni][nj]] = visited[i][j][0] + 1
                        que.append((ni,nj,field[ni][nj]))
    return -1

print(bfs())