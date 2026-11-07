# TokenMeter — Technical Debt & Performance Ledger

Source crosscheck: 2026-11-08 WIB. The locations and removed lines match current source. Diffs are independent proposals, not applied changes or measured speedups.

---

## 1. Repeated Byte-Buffer Tail Copying on Multi-Line SSE Chunks in `SSEStreamScanner.feed()`

- **Location**: `tokenmeter/proxy.py` (lines `320-326`)
- **Category**: Streaming Allocation & CPU Efficiency
- **Severity**: Medium

### Technical Analysis & Impact
`SSEStreamScanner.feed()` (`tokenmeter/proxy.py:320-326`) loops `while b'\n' in self.buffer:` and calls `self.buffer.split(b'\n', 1)` on each line. When an upstream chunk contains $K$ coalesced SSE frames, repeatedly splitting off the head copies the remaining tail of `self.buffer` $K$ times ($O(K \cdot |\text{buffer}|)$ bytes copied). Splitting once via `parts = self.buffer.split(b'\n')` and popping the incomplete tail processes the entire chunk in $O(|\text{buffer}|)$.

### Proposed Patch (`diff`)
```diff
--- a/tokenmeter/proxy.py
+++ b/tokenmeter/proxy.py
@@ -321,8 +321,11 @@
         if not chunk:
             return
         self.buffer += chunk
-        while b"\n" in self.buffer:
-            raw_line, self.buffer = self.buffer.split(b"\n", 1)
+        if b"\n" not in self.buffer:
+            return
+        parts = self.buffer.split(b"\n")
+        self.buffer = parts.pop()
+        for raw_line in parts:
             self._process_line(raw_line)
 
     def flush(self) -> None:
```

### Verification
Run `C:\Python314\python.exe -m pytest -q`.

---

## 2. Repeated Wildcard Candidate List Reconstruction & Sorting on Every `resolve_model_pricing()` Call

- **Location**: `tokenmeter/prices.py` (lines `99-108`)
- **Category**: Algorithmic Hot-Path Overhead
- **Severity**: Medium

### Technical Analysis & Impact
`resolve_model_pricing()` (`tokenmeter/prices.py:99-108`) allocates a `wildcard_candidates` list, computes `pattern.replace('*', '')`, and runs `.sort()` on every single wildcard lookup—including inside `_sync_version_and_reprice` across thousands of historical database rows. Tracking `best_rank` in a single pass eliminates the list allocation and $O(W \log W)$ sort on every call.

### Proposed Patch (`diff`)
```diff
--- a/tokenmeter/prices.py
+++ b/tokenmeter/prices.py
@@ -96,16 +96,16 @@
     if isinstance(exact, Mapping):
         return exact
 
-    wildcard_candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []
+    best_spec: Mapping[str, Any] | None = None
+    best_rank: tuple[int, int] = (-1, -1_000_000)
     for pattern, spec in models.items():
         if isinstance(pattern, str) and "*" in pattern and isinstance(spec, Mapping):
-            literal_len = len(pattern.replace("*", ""))
-            wildcard_candidates.append((literal_len, -pattern.count("*"), pattern, spec))
-
-    wildcard_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
-    for _, _, pattern, spec in wildcard_candidates:
-        if fnmatch.fnmatchcase(model, pattern):
-            return spec
+            rank = (len(pattern.replace("*", "")), -pattern.count("*"))
+            if rank > best_rank and fnmatch.fnmatchcase(model, pattern):
+                best_rank = rank
+                best_spec = spec
+    if best_spec is not None:
+        return best_spec
     return None
 
 
```

### Verification
Run `C:\Python314\python.exe -m pytest -q`.

The SSE change improves coalesced-line scanning, but a very long line fragmented across many chunks still repeatedly concatenates immutable bytes. It is not a full bounded-memory streaming redesign. Wildcard ties must retain the first matching insertion-order rule, as the strict greater-than rank comparison does.
