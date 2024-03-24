# 1700_멀티탭 스케줄링_multitap scheduling
import sys
input = sys.stdin.readline

N, K = map(int, input().split())
appliances = list(map(int, input().split()))

# 사용중 기기
using_app = set()
cnt = 0
ans = 0

for i in range(K):
    tmp = appliances[i]
    if appliances[i] in using_app:
        continue
    else:
        if cnt != N:
            using_app.add(tmp)
            cnt += 1
        else:
            # 사용중 기기 복사본
            tmp_set = set(k for k in using_app)
            for j in range(i+1, K):
                # 우선순위가 가장 낮은 기기 한개면 stop
                if len(tmp_set) == 1:
                    break
                if appliances[j] in tmp_set:
                    tmp_set.remove(appliances[j])
                    
            change = tmp_set.pop()
            using_app.remove(change)
            using_app.add(tmp)
            ans += 1

print(ans)

# 더 빠른? 코드
"""
# 1700_멀티탭 스케줄링_multitap scheduling
import sys
# import time
input = sys.stdin.readline
# start_time = time.time()

N, K = map(int, input().split())
appliances = list(map(int, input().split()))

using_app = set()
cnt = 0
ans = 0

next_use = {}

for i in range(K):
    if appliances[i] not in next_use.keys():
        next_use[appliances[i]] = []
    else:
        next_use[appliances[i]].append(i)


for i in range(K):
    tmp = appliances[i]
    if appliances[i] in using_app:
        next_use[tmp].pop(0)
        continue
    else:
        if cnt != N:
            using_app.add(tmp)
            cnt += 1
        else:
            change = [-1, -1]
            for s in using_app:
                if not next_use[s]:
                    using_app.remove(s)
                    using_app.add(tmp)
                    break
                else:
                    if next_use[s][0] > change[1]:
                        change = [s, next_use[s][0]]
            else:
                next_use[change[0]].pop(0)
                using_app.remove(change[0])
                using_app.add(tmp)
            ans += 1

print(ans)
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
"""