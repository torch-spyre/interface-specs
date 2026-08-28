# coreIdToDscSchedule Field: SuperDsc Class Documentation

## Definition

**Location**: [superdsc.h]

```cpp
std::map<int, std::vector<DscScheduleStep>> coreIdToDscSchedule;
```

**Type**: Map from core ID (int) to vector of DscScheduleStep objects

---

## DscScheduleStep Structure

**Location**: [superdsc.h]

```cpp
struct DscScheduleStep {
  int datadsc_idx = -1;   // Index of DataOpDsc (-1 if no data op)
  int dldsc_idx = -1;     // Index of DLDSC (DeepLearning DSC) (-1 if no DL op)

  bool before_sync = false;  // Sync before this step
  bool after_sync = false;   // Sync after this step

  DscScheduleStep(int datadsc_idx, int dldsc_idx, bool before_sync, bool after_sync)
      : datadsc_idx(datadsc_idx),
        dldsc_idx(dldsc_idx),
        before_sync(before_sync),
        after_sync(after_sync) {}
};
```

---

## Purpose

The **`coreIdToDscSchedule`** field defines the **execution sequence** of operations on each core by specifying:

1. **Which DSCs** (DesignSpaceConfigs) execute on this core
2. **The order** they execute in (sequence of steps)
3. **Data flow dependencies** (before_sync / after_sync for synchronization)

### High-Level Purpose

Maps each core to its **schedule of operations**, telling the system:
- When does this core run a DataOp (data transfer)?
- When does this core run a DL operation (compute)?
- What synchronization barriers exist between operations?

---

## JSON Export Format

### Example from sdsc.out.out.json (Line 36-38)

```json
"coreIdToDscSchedule" : {
  "0" : [[-1 ,0 ,0 ,0]]
}
```

**Breakdown**:
- **Key**: `"0"` → Core ID = 0
- **Value**: `[[-1, 0, 0, 0]]` → Single DscScheduleStep for this core
  - **Position 0** (-1): datadsc_idx = -1 (no DataOp)
  - **Position 1** (0): dldsc_idx = 0 (use DLDSC at index 0)
  - **Position 2** (0): before_sync = false
  - **Position 3** (0): after_sync = false

### Multi-Step Example

```json
"coreIdToDscSchedule" : {
  "0" : [
    [0, -1, 0, 1],    // Step 0: DataOp 0, then sync after
    [-1, 0, 1, 0],    // Step 1: Sync before, then DL op 0
    [1, -1, 0, 0]     // Step 2: DataOp 1, no sync
  ],
  "1" : [
    [0, -1, 0, 1],    // Same sequence for core 1
    [-1, 0, 1, 0],
    [1, -1, 0, 0]
  ]
}
```

## Field Interpretation

### datadsc_idx

- **-1**: No data operation in this step
- **0+**: Index of DataOpDsc to execute (data transfer, shuffle, etc.)

### dldsc_idx

- **-1**: No compute operation in this step
- **0+**: Index of DesignSpaceConfig (compute) to execute

### before_sync / after_sync

- **true**: Insert synchronization barrier before/after this step
- **false**: No synchronization needed
- **Purpose**: Coordinate between cores when data dependencies exist

### Valid Combinations

| datadsc_idx | dldsc_idx | Meaning |
|-------------|-----------|---------|
| -1 | ≥0 | Execute compute DSC only |
| ≥0 | -1 | Execute data op only |
| -1 | -1 | Invalid (should not occur) |
| ≥0 | ≥0 | Both data and compute in one step (rare) |

---

## Execution Model

### Single-DSC Single-Core (Simple)

```
Core 0: [[-1, 0, 0, 0]]
        └─ One step: no data op, DL DSC 0, no syncs
           → Core 0 runs compute operation on DSC 0
```

### Multi-DSC Multi-Core (Complex)

```
Core 0: [[0, -1, 0, 1],    // Step 1: DataOp 0, sync after
         [-1, 0, 1, 0]]    // Step 2: Sync before, compute DSC 0

Core 1: [[0, -1, 0, 1],    // Same sequence
         [-1, 0, 1, 0]]
```

**Execution Sequence**:
1. Both cores: Execute DataOp 0 (data transfer)
2. Both cores: Sync barrier (after_sync from step 1)
3. Both cores: Sync barrier (before_sync in step 2)
4. Both cores: Execute compute on DSC 0

---

## Constraint from Code Comments (L3DlOpsScheduler.cpp:363-365)

```cpp
// For a given DSC, any cores that it uses must have the same information in
// their DscScheduleStep. Therefore, we only need to look at one of the
// coreIds. Pick the first coreId.
```

**Implication**: All cores using the same DSC follow identical schedules. This simplifies the L3 scheduler's planning—it can look at just one representative core to understand the full schedule.

---

## Key Insights

### 1. Core → Operations Mapping

Instead of asking "which operations does DSC 0 run?", the system asks "what operations does core 0 run?" The answer is a sequence of steps specifying DSC indices.

### 2. Coordination via Sync Flags

The `before_sync` and `after_sync` flags enable the L3 scheduler to insert memory barriers exactly where data dependencies require them, without unnecessary barriers.

### 3. Simple Export Format

Reducing to a 4-element array `[datadsc_idx, dldsc_idx, before_sync, after_sync]` keeps the JSON compact and easy to parse, while preserving all scheduling information.

### 4. One Representative Core

Since all cores using the same DSC follow identical schedules, the system only needs to store one copy of the schedule per core, reducing redundancy.

---

