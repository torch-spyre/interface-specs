# SDSC Bundle API Documentation

---

## Table of Contents

Documents are organized in the recommended reading order, grouped by learning stage.

---

### Stage 1 — Introduction

> Understand the Spyre stack and the two API components (MLIR + JSON) before diving into either layer.

| # | Section | File |
|---|---------|------|
| 1.1 | Overview | [Overview.md](Overview.md) |

---

### Stage 2 — MLIR Bundle (Orchestration Layer)

> The `.mlir` file that orchestrates execution flow, symbol management, and operation sequencing across one or more SDSC JSON files.

| # | Section | File |
|---|---------|------|
| 2.1 | MLIR Bundle API | [MLIR-bundle-API.md](MLIR-bundle-API.md) |
| 2.2 | Bundle Usage Examples | [MLIR-bundle-usage-examples.md](MLIR-bundle-usage-examples.md) |
| 2.3 | Complete Example (MLIR) | [MLIR-complete-example.md](MLIR-complete-example.md) |

---

### Stage 3 — JSON Layer Fundamentals

> The overall JSON file structure and complete object map; read before tackling individual object specs.

| # | Section | File |
|---|---------|------|
| 3.1 | SDSC JSON API | [SDSC-json-api.md](SDSC-json-api.md) |
| 3.2 | Object Hierarchy | [JSON-object-Hierarchy.md](JSON-object-Hierarchy.md) |

---

### Stage 4 — Core Concepts (Foundational Objects)

> Folding, padding, and stick layout are foundational; every later object references these ideas.

| # | Section | File |
|---|---------|------|
| 4.1 | SuperDsc Object | [superdsc-object.md](superdsc-object.md) |
| 4.2 | FoldProperty | [foldproperty.md](foldproperty.md) |
| 4.3 | FoldManager | [foldmanager.md](foldmanager.md) |
| 4.4 | Padding | [padding.md](padding.md) |
| 4.5 | Stick Layout Constraints | [stick-layout-constraints.md](stick-layout-constraints.md) |

---

### Stage 5 — DesignSpaceConfig & Tensor Objects

> The `dscs_[]` array and the tensor-descriptor objects that make up the bulk of every real SDSC JSON file.

| # | Section | File |
|---|---------|------|
| 5.1 | DesignSpaceConfig | [designspaceconfig.md](designspaceconfig.md) |
| 5.2 | LabeledDataStructure | [labeleddatastructure.md](labeleddatastructure.md) |
| 5.3 | MemoryOrganization | [memoryorganization.md](memoryorganization.md) |
| 5.4 | PrimaryDsInfo | [primarydsinfo.md](primarydsinfo.md) |
| 5.5 | DataStructDims | [datastructdims.md](datastructdims.md) |

---

### Stage 6 — Execution & Scheduling Objects

> Per-core data staging, the schedule tree, coordinate mappings, compute operations, and constant tensors.

| # | Section | File |
|---|---------|------|
| 6.1 | DataStageParam | [datastageparam.md](datastageparam.md) |
| 6.2 | ScheduleTreeNode | [scheduletreenode.md](scheduletreenode.md) |
| 6.3 | CoordinateContainer | [coordinatecontainer.md](coordinatecontainer.md) |
| 6.4 | CoordinateInfo | [coordinateinfo.md](coordinateinfo.md) |
| 6.5 | ComputeOperation | [computeoperation.md](computeoperation.md) |
| 6.6 | ConstantInfo | [constantinfo.md](constantinfo.md) |

---

### Stage 7 — Worked Examples

> End-to-end examples that tie all objects together; read after completing Stage 4.

| # | Section | File |
|---|---------|------|
| 7.1 | Complete Example (JSON) | [complete-example.md](complete-example.md) |

---

### Stage 8 — Production Readiness

> Error handling, authoring best practices, and the reference glossary.

| # | Section | File |
|---|---------|------|
| 8.1 | Error Handling | [error-handling.md](error-handling.md) |
| 8.2 | Best Practices | [best-practices.md](best-practices.md) |
| 8.3 | Reference Documents & Glossary | [reference.md](reference.md) |

---
