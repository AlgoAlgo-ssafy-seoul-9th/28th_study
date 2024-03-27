N, K = map(int, input().split())
elec = list(map(int, input().split()))
multi = [0]*(K+1)           # 실제 크기가 아니라 카운트배열로 선언
elec_cnt = 0                # 현재 연결된 기기 수
ans = 0                     # 기기를 뽑은 횟수
arr = [[0]*(K+1) for _ in range(K+1)]   # 얼마나 뒤에 사용되는지 표시
for j in range(K-1, 0, -1):
    for i in range(1, K+1):
        if i == elec[j]:
            arr[i][j-1] = 1
        elif arr[i][j]:
            arr[i][j-1] = arr[i][j] + 1

for j in range(K):
    if elec_cnt<N and multi[elec[j]]==0:    # 남은 자리가 있고, 처음 사용되는 기기면
        elec_cnt += 1
        multi[elec[j]] = 1
    elif elec_cnt==N and multi[elec[j]]==0: # 자리가 없고, 새로운 기기면, 가장 나중에 사용될 기기를 제거
        max_idx = 0
        for i in range(1, K+1):
            if multi[i] and arr[i][j]==0:   # 다시 사용하지 않는 기기면 제거
                max_idx = i
                break
            if multi[i] and arr[max_idx][j] < arr[i][j]:
                max_idx = i
        multi[max_idx] = 0
        multi[elec[j]] = 1
        ans += 1
print(ans)
