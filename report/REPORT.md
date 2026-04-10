# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hải Đăng
**Nhóm:** 71
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)
**High cosine similarity nghĩa là gì?**
Hai đoạn văn có high cosine similarity nghĩa là chúng có nội dung/ngữ nghĩa gần giống nhau, dù có thể dùng từ khác nhau. Vector embedding của chúng “cùng hướng” trong không gian vector.

**Ví dụ HIGH similarity:**
- Sentence A: Python is a popular programming language for data science.
- Sentence B: Python is a popular programming language for data science.
- Tại sao tương đồng:
Cả hai đều nói về Python và ứng dụng trong data science → cùng chủ đề → embedding gần nhau.
**Ví dụ LOW similarity:**
- Sentence A: I love eating pizza on weekends.
- Sentence B: Quantum mechanics studies subatomic particles.
- Tại sao khác:
Hai câu thuộc hai domain hoàn toàn khác nhau (food vs physics) → embedding khác hướng.
**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Vì cosine similarity chỉ quan tâm đến hướng (ngữ nghĩa), không bị ảnh hưởng bởi độ dài vector. Trong khi Euclidean distance bị ảnh hưởng bởi magnitude → không phù hợp với embedding.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / (500 - 50))
*Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
- Khi overlap tăng → bước nhảy nhỏ hơn → số chunks tăng lên.
- Overlap nhiều giúp giữ context giữa các chunk, tránh mất thông tin ở ranh giới.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** CV tìm việc

**Tại sao nhóm chọn domain này?**
Nhóm lựa chọn domain CV cá nhân vì dữ liệu có cấu trúc rõ ràng với các phần quen thuộc như Education, Experience, Skills, Projects, giúp dễ dàng thiết kế các truy vấn benchmark có mục tiêu cụ thể. Ngoài ra, bài toán này phản ánh trực tiếp một use case thực tế trong lĩnh vực HR/Recruitment — nơi RAG được ứng dụng để tìm kiếm và so sánh ứng viên.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | LongCV.md | data/group_dataset/ | 7,286 | candidate_name, position=AI Engineer, years_experience=2, university=FPT, gpa=3.2 |
| 2 | PLinhCV.md | data/group_dataset/ | 3,409 | candidate_name, position=Full-stack Developer, years_experience=5, university=FPT, gpa=8.4 |
| 3 | DangCV.md | data/group_dataset/ | 2,638 | candidate_name, position=AI Engineer, years_experience=1, university=FPT, gpa=3.04 |
| 4 | PhuCV.md | data/group_dataset/ | 2,204 | candidate_name, position=DevOps Engineer, years_experience=1, university=FPT, gpa=3.4 |
| 5 | TLinhCV.md | data/group_dataset/ | 4,288 | candidate_name, position=Business Administration, years_experience=0, university=NEU, gpa=3.85 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `candidate_name` | str | `"Ngo Van Long"` | Truy xuất kết quả theo tên ứng viên cụ thể |
| `position` | str | `"AI Engineer"` | Filter để chỉ tìm trong nhóm vị trí phù hợp (e.g. chỉ DevOps) |
| `years_experience` | int | `2` | Phân biệt junior/senior khi cần filter theo kinh nghiệm |
| `university` | str | `"FPT University"` | Filter theo trường nếu tuyển dụng yêu cầu trường cụ thể |
| `gpa` | str | `"3.2/4.0"` | Thêm ngữ cảnh học lực vào kết quả trả về |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| CV_NGOVANLONG (7,286 chars) | FixedSizeChunker | 29 | 300 chars | Không — cắt giữa câu |
| CV_NGOVANLONG (7,286 chars) | SentenceChunker | 14 | 515 chars | Có — giữ trọn câu |
| CV_Phuong Linh (3,409 chars) | FixedSizeChunker | 14 | 290 chars | Không — cắt giữa câu |
| CV_Phuong Linh (3,409 chars) | SentenceChunker | 4 | 849 chars | Có — nhưng chunk quá lớn |
| CV_Phuong Linh (3,409 chars) | RecursiveChunker | 15 | 225 chars | Tốt — chunk nhỏ, cân bằng |
| CV_Hai Dang (2,638 chars) | FixedSizeChunker | 11 | 285 chars | Không — cắt giữa câu |
| CV_Hai Dang (2,638 chars) | SentenceChunker | 1 | 2,636 chars | Không — toàn bộ CV là 1 chunk |
| CV_Hai Dang (2,638 chars) | RecursiveChunker | 11 | 238 chars | Tốt — chia hợp lý |
### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
Văn bản được tách dựa trên ranh giới câu (sau các dấu ., !, ? hoặc xuống dòng), sau đó gộp mỗi 2 câu thành một chunk. Do nội dung CV thường gồm các câu ngắn hoặc bullet mô tả thành tích, việc gom 2 câu giúp mỗi chunk giữ được một ý nghĩa tương đối trọn vẹn mà vẫn đảm bảo độ dài phù hợp.

**Tại sao tôi chọn strategy này cho domain nhóm?**
CV thường được viết dưới dạng bullet points, trong đó mỗi bullet tương ứng với 1–2 câu mô tả kỹ năng hoặc kinh nghiệm cụ thể. SentenceChunker tận dụng tốt cấu trúc này bằng cách giữ nguyên các ranh giới ngữ nghĩa tự nhiên. Tuy nhiên, một hạn chế là với các CV sử dụng bảng Markdown, nội dung có thể bị gộp thành một chunk lớn duy nhất — đây là một failure case cần được ghi nhận.

**Code snippet (nếu custom):**
```python
from src.chunking import SentenceChunker
from src.models import Document
from src.store import EmbeddingStore

chunker = SentenceChunker(max_sentences_per_chunk=2)
store = EmbeddingStore(collection_name="cv_store")

for doc in cv_docs:
    chunks = chunker.chunk(doc.content)
    chunked_docs = [
        Document(
            id=f"{doc.id}_chunk_{j}",
            content=chunk,
            metadata={**doc.metadata, "chunk_index": j}
        )
        for j, chunk in enumerate(chunks)
    ]
    store.add_documents(chunked_docs)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| DangCV | RecursiveChunker (best baseline) | 11 | 238 chars | Tốt — chia theo cấu trúc, chunk cân bằng |
| DangCV | *SentenceChunker (của tôi) | 1 | 2636 chars | Kém — mất khả năng chia nhỏ, khó match |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đăng (tôi) | SentenceChunker(max=2) | 14 | Giữ ngữ cảnh câu, phù hợp bullet CV | Thất bại với CV dùng bảng/ít dấu câu |
| Long | FixedSizeChunker(300, overlap=50) | 29 | Đơn giản, dễ kiểm soát size | Cắt giữa câu, mất ngữ nghĩa |
| Phương Linh | RecursiveChunker(400) | ~20 | Cân bằng, robust với mọi format | Chunk size không đồng đều |
| Phú | CVSectionChunker (by ## heading) | ~8-10 | Chunk đúng theo section CV | Phụ thuộc CV có heading rõ ràng |


**Strategy nào tốt nhất cho domain này? Tại sao?**
CVSectionChunker của Phú là lựa chọn phù hợp nhất về mặt lý thuyết, do CV dạng Markdown thường có cấu trúc rõ ràng với các heading như ## Education, ## Skills. Mỗi section tương ứng với một đơn vị ngữ nghĩa hoàn chỉnh, giúp quá trình retrieval tránh bị cắt ngang thông tin quan trọng.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
Em split text thành các câu bằng cách dựa vào dấu câu như ., !, ? (regex đơn giản). Sau đó group các câu lại thành chunk với số lượng tối đa max_sentences_per_chunk.
Edge case: xử lý khoảng trắng dư và đảm bảo mỗi chunk là string hợp lệ.

**`RecursiveChunker.chunk` / `_split`** — approach:
Em implement theo chiến lược chia nhỏ dần: thử split bằng các separator lớn (ví dụ \n\n, \n, ., space). Nếu chunk vẫn quá lớn thì tiếp tục split đệ quy với separator nhỏ hơn.
Base case là khi chunk nhỏ hơn chunk_size hoặc không còn separator → fallback sang fixed-size chunking.

### EmbeddingStore

**`add_documents` + `search`** — approach:
Em lưu trữ dữ liệu dưới dạng list các dictionary gồm id, content, metadata, và embedding. Khi thêm document, tôi gọi embedding function để tạo vector và lưu lại.
Khi search, tôi embed query rồi tính cosine similarity với từng document, sau đó sort giảm dần theo score và trả về top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
Với filtering, em lọc documents theo metadata trước, sau đó mới tính similarity để giảm chi phí tính toán.
Với delete, tôi remove tất cả records có id trùng với doc_id bằng cách rebuild lại list store.

### KnowledgeBaseAgent

**`answer`** — approach:
Em thực hiện retrieval trước bằng cách gọi store.search() để lấy top-k relevant chunks. Sau đó xây dựng prompt gồm context + question.
Context được nối từ các chunks, sau đó đưa vào LLM để generate câu trả lời dựa trên thông tin đã retrieve.

### Test Results

```
# Paste output of: pytest tests/ -v
```
============================================================= test session starts =============================================================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0 -- /Users/dangnguyen/Documents/D7/2A202600157-NguyenHaiDang-Day07/venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/dangnguyen/Documents/D7/2A202600157-NguyenHaiDang-Day07
plugins: anyio-4.12.1
collected 42 items                                                                                                                            

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                   [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                            [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                     [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                      [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                           [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                           [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                 [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                  [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                  [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                  [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                             [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                         [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                   [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                          [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                              [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                        [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                              [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                  [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                    [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                      [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                            [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                 [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                   [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                       [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                    [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                             [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                            [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                       [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                   [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                              [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                  [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                        [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                  [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                               [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                             [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                            [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                           [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                    [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                          [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                              [100%]

============================================================= 42 passed in 0.04s ==============================================================

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1    | Python is used for data science. | Python is popular in machine learning. | high    | 0.89         ||
| 2    | I love eating pizza.             | Quantum physics studies atoms.         | low     | 0.05         ||
| 3    | Dogs are loyal animals.          | Cats are independent pets.             | high    | 0.72         ||
| 4    | The sky is blue.                 | Blue color is calming.                 | high    | 0.65         ||
| 5    | Programming in Java is powerful. | I enjoy swimming in the ocean.         | low     | 0.12         ||


**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Cặp (3) và (4) là đáng chú ý nhất, vì dù không đồng nghĩa hoàn toàn, mô hình vẫn cho điểm tương đồng khá cao (0.65–0.72). Điều này cho thấy embeddings không chỉ dựa trên từ khóa giống nhau mà còn nắm bắt được ngữ nghĩa liên quan (semantic similarity), ví dụ cùng thuộc một chủ đề hoặc ngữ cảnh gần nhau. Đồng thời, kết quả cũng cho thấy embeddings có xu hướng “làm mịn” nghĩa, tức là các câu khác nhau nhưng cùng domain vẫn có thể được xem là tương tự ở mức độ nhất định.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Ai từng làm Data Engineer tại FPT Software và đạt hiệu suất cải thiện bao nhiêu %? | Nguyễn Phương Linh, cải thiện ~30% thời gian xử lý | FPT Software / Data Engineer (Phương Linh CV) |
| 2 | Sinh viên nào đạt giải nhất Hackathon nội bộ và thiết kế giải pháp AI? | Ngô Văn Long | Innovation & Solution Design (Văn Long CV) |
| 3 | Ai có kinh nghiệm triển khai trên Kubernetes (GKE) và tham gia Huawei Seeds of The Future 2024? | Nguyễn Mạnh Phú | Professional Summary / Awards (Mạnh Phú CV) |
| 4 | Trong các dự án iOS, ai đã tích hợp RAG và đó là dự án nào? | Nguyễn Hải Đăng, dự án "Poems Trading App" | Junior IOS Developer (Hải Đăng CV) |
| 5 | Ai giữ vai trò Finance & Business Lead cho dự án LifeTrack? | Nguyễn Thùy Linh | Work & Project Experience (Thùy Linh CV) |

### Kết Quả Của Tôi

**Q1 — Data Engineer FPT Software** (gold: Nguyễn Phương Linh)

| Rank | Ứng viên           | Position             | Score  | Relevant? |
| ---- | ------------------ | -------------------- | ------ | --------- |
| #1   | Nguyen Phuong Linh | Full-stack Developer | 0.3412 | ✅         |
| #2   | Ngo Van Long       | AI Engineer          | 0.3361 | ❌         |
| #3   | Ngo Van Long       | AI Engineer          | 0.3078 | ❌         |

**Q2 — Giải nhất Hackathon nội bộ** (gold: Ngô Văn Long)

| Rank | Ứng viên           | Position             | Score  | Relevant?    |
| ---- | ------------------ | -------------------- | ------ | ------------ |
| #1   | Nguyen Hai Dang    | AI Engineer          | 0.3420 | ❌            |
| #2   | Ngo Van Long       | AI Engineer          | 0.2715 | ⚠️ (partial) |
| #3   | Nguyen Phuong Linh | Full-stack Developer | 0.2714 | ❌            |


**Q3 — Kubernetes GKE + Huawei Seeds** (gold: Nguyễn Mạnh Phú)

| Rank | Ứng viên        | Position        | Score  | Relevant? |
| ---- | --------------- | --------------- | ------ | --------- |
| #1   | Nguyen Manh Phu | DevOps Engineer | 0.1888 | ✅         |
| #2   | Nguyen Manh Phu | DevOps Engineer | 0.1239 | ✅         |
| #3   | Nguyen Manh Phu | DevOps Engineer | 0.0876 | ✅         |


**Q4 — RAG trong dự án iOS**  (gold: Nguyễn Hải Đăng)

| Rank | Ứng viên        | Position        | Score  | Relevant?                          |
| ---- | --------------- | --------------- | ------ | ---------------------------------- |
| #1   | Nguyen Manh Phu | DevOps Engineer | 0.3443 | ❌                                  |
| #2   | Ngo Van Long    | AI Engineer     | 0.3309 | ❌                                  |
| #3   | Nguyen Hai Dang | AI Engineer     | 0.3225 | ⚠️ (partial, nhưng chưa match RAG) |


**Q5 — Finance & Business Lead LifeTrack**  (gold: Nguyễn Thùy Linh)

| Rank | Ứng viên         | Position         | Score  | Relevant? |
| ---- | ---------------- | ---------------- | ------ | --------- |
| #1   | Nguyen Thuy Linh | Business Analyst | 0.3298 | ✅         |
| #2   | Nguyen Hai Dang  | AI Engineer      | 0.3204 | ❌         |
| #3   | Nguyen Manh Phu  | DevOps Engineer  | 0.3145 | ❌         |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Tôi học được rằng việc tận dụng cấu trúc tài liệu (như dùng CVSectionChunker) có thể giúp giữ nguyên ngữ nghĩa tốt hơn so với chỉ dựa vào độ dài chunk. Cách tiếp cận này đặc biệt hiệu quả với dữ liệu có format rõ ràng như CV Markdown, nơi mỗi section mang một ý nghĩa riêng biệt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Tôi nhận thấy một số nhóm đã sử dụng hybrid strategy (kết hợp nhiều cách chunking) để xử lý các loại dữ liệu khác nhau trong cùng một hệ thống. Điều này giúp tăng độ linh hoạt và cải thiện retrieval quality, thay vì phụ thuộc hoàn toàn vào một strategy duy nhất.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Nếu làm lại, tôi sẽ ưu tiên sử dụng RecursiveChunker hoặc kết hợp với rule-based chunking theo cấu trúc (heading, bullet). Ngoài ra, tôi cũng sẽ cải thiện metadata (ví dụ thêm project_name, skills) để hỗ trợ filtering tốt hơn, từ đó tăng độ chính xác của retrieval.
---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14/ 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 9/ 10 |
| Core implementation (tests) | Cá nhân | 30/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | ** 87/ 100** |
