N = int(input())
A = [0]*(N+1)   # 행열크기
for i in range(N):
    A[i], A[i+1] = map(int, input().split())  # 행렬 크기 저장

D = [[0]*(N+1) for _ in range(N+1)]            # Dij : Ai부터 Aj까지 최소 곱셈횟수

for l in range(1, N):                          # 곱하는 행렬의 개수
    for i in range(1, N-l+1):                  # i 곱하는 맨 앞 행렬
        j = i+l                                # j 맨 마지막 행렬 
        min_v = 1000000000
        for k in range(i, j):                  # (Ai...Ak)(Ak+1...Aj) 결합법칙 적용하는 왼쪽 괄호의 끝 행렬번호
            min_v = min(min_v, D[i][k]+D[k+1][j]+A[i-1]*A[k]*A[j])     # 결합 위치를 바꿨을 때 최소 곱셈 횟수 갱신
        D[i][j] = min_v                        #  Ai...Aj까지 최소 곱셈 횟수

print(D[1][N])
