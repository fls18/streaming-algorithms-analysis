import csv
import sys
import time
import hashlib
import random
import math

# 1. 스트리밍 데이터 로더 정의 
def stream_dataset(file_path):
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stock_code = row.get('StockCode', '').strip()
            if stock_code:
                yield stock_code

# 2. 알고리즘 4종 구현 
# [1] Bloom Filter
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size

    def _hashes(self, item):
        indices = []
        for i in range(self.num_hashes):
            hash_string = f"{item}-{i}".encode('utf-8')
            hash_val = int(hashlib.md5(hash_string).hexdigest(), 16)
            indices.append(hash_val % self.size)
        return indices

    def add(self, item):
        for index in self._hashes(item):
            self.bit_array[index] = 1

    def check(self, item):
        for index in self._hashes(item):
            if self.bit_array[index] == 0:
                return False
        return True


# [2] Count-Min Sketch
class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def _hashes(self, item):
        indices = []
        for i in range(self.depth):
            hash_string = f"{item}-{i}".encode('utf-8')
            hash_val = int(hashlib.sha256(hash_string).hexdigest(), 16)
            indices.append(hash_val % self.width)
        return indices

    def add(self, item):
        for row, col in enumerate(self._hashes(item)):
            self.table[row][col] += 1

    def estimate(self, item):
        return min(self.table[row][col] for row, col in enumerate(self._hashes(item)))


# [3] HyperLogLog 
class HyperLogLog:
    def __init__(self, p):
        self.p = p                          # 등급 파라미터 (버킷 개수 결정)
        self.m = 1 << p                     # 버킷 개수 (2^p)
        self.registers = [0] * self.m       # 레지스터 배열
        # 버킷 개수에 따른 알파 상수 보정값
        if self.m == 16: self.alpha = 0.673
        elif self.m == 32: self.alpha = 0.697
        elif self.m == 64: self.alpha = 0.709
        else: self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def add(self, item):
        # 32비트 해시 생성
        hash_str = hashlib.sha1(item.encode('utf-8')).hexdigest()
        hash_val = int(hash_str[:8], 16)

        # 앞의 p개 비트는 버킷 인덱스로 사용
        idx = hash_val & (self.m - 1)
        # 나머지 비트에서 처음으로 1이 나타나는 위치(연속된 0의 개수 + 1) 계산
        w = hash_val >> self.p
        rho = 1
        while w > 0 and (w & 1) == 0:
            rho += 1
            w >>= 1

        # 레지스터 최댓값 갱신
        self.registers[idx] = max(self.registers[idx], rho)

    def estimate(self):
        # 조화 평균 계산
        est = self.alpha * (self.m ** 2) / sum(2.0 ** -r for r in self.registers)

        # 선형 카운팅 보정 (소량의 데이터 처리 시)
        if est <= 2.5 * self.m:
            v = self.registers.count(0)
            if v > 0:
                est = self.m * math.log(self.m / v)
        return int(est)


# [4] Reservoir Sampling
class ReservoirSampling:
    def __init__(self, k):
        self.k = k          # 샘플 사이즈
        self.count = 0      # 처리한 총 데이터 수
        self.reservoir = [] # 샘플 저장소

    def add(self, item):
        self.count += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # 확률 k/N으로 기존 샘플을 새 데이터로 교체
            i = random.randint(0, self.count - 1)
            if i < self.k:
                self.reservoir[i] = item


# 3. 메모리 측정 유틸리티 함수
def get_memory_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, list):
        size += sum(sys.getsizeof(item) for item in obj)
        if obj and isinstance(obj[0], list):
            for sublist in obj:
                size += sum(sys.getsizeof(item) for item in sublist)
    elif isinstance(obj, dict):
        size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
    elif isinstance(obj, set):
        size += sum(sys.getsizeof(item) for item in obj)
    return size


# 4. 통합 테스트 벤치마크 런너 
def run_all_experiments(file_path, bf_size=100000, bf_hashes=3, cms_width=10000, cms_depth=5, hll_p=10, rs_k=500):
    # 알고리즘 초기화
    bf = BloomFilter(size=bf_size, num_hashes=bf_hashes)
    cms = CountMinSketch(width=cms_width, depth=cms_depth)
    hll = HyperLogLog(p=hll_p)
    rs = ReservoirSampling(k=rs_k)

    # Ground Truth 계산용 공간
    gt_set = set()
    gt_dict = {}
    gt_all_list = [] # 샘플 분포 대조용 원본 데이터 전체 기록

    print(f" 데이터 주입 중... (BF: size={bf_size}, CMS: width={cms_width}, HLL: buckets={1<<hll_p}, RS: k={rs_k})")

    total_records = 0
    for item in stream_dataset(file_path):
        total_records += 1
        bf.add(item)
        cms.add(item)
        hll.add(item)
        rs.add(item)

        # Ground Truth 업데이트
        gt_set.add(item)
        gt_dict[item] = gt_dict.get(item, 0) + 1
        gt_all_list.append(item)

    # 1️ Bloom Filter 검증 (FPR)
    fp_count = 0
    for i in range(10000):
        if bf.check(f"FAKE_ITEM_{i}"): fp_count += 1
    fpr = (fp_count / 10000) * 100

    # 2️ Count-Min Sketch 검증 (MRE)
    total_error = 0
    for item, true_count in gt_dict.items():
        total_error += (cms.estimate(item) - true_count)
    mre = total_error / len(gt_dict)

    # 3️ HyperLogLog 검증 (정확도 오차)
    true_unique = len(gt_set)
    hll_est = hll.estimate()
    hll_error = abs(hll_est - true_unique) / true_unique * 100

    # 4️ Reservoir Sampling 검증 (샘플링 신뢰성)
    # 저장된 샘플들이 실제 전체 데이터 분포 상위권에 고르게 속해 있는지 간접 확인
    sample_in_gt = sum(1 for x in rs.reservoir if x in gt_set)
    rs_match_rate = (sample_in_gt / len(rs.reservoir)) * 100 if rs.reservoir else 0

    print("\n ────────── [실험 결과 리포트] ──────────")
    print(f"· 총 처리 스트림 수: {total_records:,} 개")
    print(f"· 실제 고유 아이템 수: {true_unique:,} 개")
    print("--------------------------------------------------")
    print(f"① Bloom Filter")
    print(f"  - False Positive Rate (FPR): {fpr:.4f}% (FN은 메커니즘상 0%)")
    print(f"  - 메모리: 알고리즘 {get_memory_size(bf.bit_array)/1024:.2f} KB vs Ground Truth {get_memory_size(gt_set)/1024:.2f} KB")
    print(f"② Count-Min Sketch")
    print(f"  - 평균 과다추정 오차 (MRE): {mre:.4f}")
    print(f"  - 메모리: 알고리즘 {get_memory_size(cms.table)/1024:.2f} KB vs Ground Truth {get_memory_size(gt_dict)/1024:.2f} KB")
    print(f"③ HyperLogLog")
    print(f"  - 추정 고유 키 수: {hll_est:,} 개 (실제값 대비 오차율: {hll_error:.2f}%)")
    print(f"  - 메모리: 알고리즘 {get_memory_size(hll.registers)/1024:.2f} KB (고정 레지스터 {1<<hll_p}개)")
    print(f"④ Reservoir Sampling")
    print(f"  - 추출된 최종 샘플 수: {len(rs.reservoir)} 개 (설정된 k={rs_k})")
    print(f"  - 샘플 유효성(Unique 대조율): {rs_match_rate:.2f}%")
    print("──────────────────────────────────────────────────\n")

# ==========================================
# 5. 파라미터 비교 실험 자동화 루프 
# ==========================================
if __name__ == "__main__":
    target_file = 'OnlineRetail.csv'

    print("🏁 4대 스트리밍 알고리즘 파라미터 변동 비교 테스트 벤치마크 시작\n")

    # [실험 1] Bloom Filter 파라미터 대조
    print(" [실험 1] Bloom Filter 변동 실험 (배열 크기 확장)")
    run_all_experiments(target_file, bf_size=50000, bf_hashes=3)
    run_all_experiments(target_file, bf_size=200000, bf_hashes=3)

    # [실험 2] Count-Min Sketch 파라미터 대조
    print(" [실험 2] Count-Min Sketch 변동 실험 (Width 가로폭 확장)")
    run_all_experiments(target_file, cms_width=5000, cms_depth=4)
    run_all_experiments(target_file, cms_width=20000, cms_depth=4)

    # [실험 3] HyperLogLog 파라미터 대조
    print(" [실험 3] HyperLogLog 변동 실험 (p값 변동에 따른 버킷 수 조절)")
    run_all_experiments(target_file, hll_p=6)   # 버킷 64개
    run_all_experiments(target_file, hll_p=12)  # 버킷 4096개

    # [실험 4] Reservoir Sampling 파라미터 대조
    print(" [실험 4] Reservoir Sampling 변동 실험 (샘플 사이즈 k 크기 변동)")
    run_all_experiments(target_file, rs_k=100)
    run_all_experiments(target_file, rs_k=1000)


