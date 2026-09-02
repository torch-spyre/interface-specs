# SDSC JSON API

## Introduction

Each `sdsc_*.json` file in a SuperDSC-Bundle describes a single torch operation to be executed on the Spyre backend (DeepTools). One JSON file encodes everything the hardware needs to execute that operation deterministically across 1 or multiple cores: how the iteration space is divided, how tensors are laid out in memory, where data lives (HBM vs. LX scratchpad), and what compute to perform.

A PyTorch model may translate into several SuperDSC-Bundles. Each bundle is composed of a `bundle.mlir` file (which orchestrates execution flow and symbol management — see [MLIR Bundle API](MLIR-bundle-API.md)) and one or more `sdsc_*.json` files. Each JSON file corresponds to one torch operation.

The different sections of an SDSC JSON file describe the following:

- **Core fold properties** — how the iteration space is divided across cores
- **Tensor descriptors** — layout, memory residency, data format, and stick configuration for each tensor
- **Schedule tree** — per-tensor memory allocation, start addresses, and coordinate mappings
- **Data staging** — per-core tile sizes for steady-state and epilogue passes
- **Compute operations** — execution unit, operation name, and input/output tensor references

## Key Components

An SDSC JSON file is structured as a single top-level key (the operation name) whose value is a **SuperDsc** object. The SuperDsc object holds a few top-level fields and a `dscs_[]` array of **DesignSpaceConfig** entries. Each `dscs_[]` entry is itself a single-key object wrapping a `DesignSpaceConfig` — some of its fields are leaf values while others are composite objects that drill down further into the object hierarchy.

![SDSC JSON File — Object Hierarchy](figures/sdsc_json_hierarchy.png)

*Figure: Complete object hierarchy of an SDSC JSON file.*

**Colour legend**

| Colour | Group | Objects |
|---|---|---|
| Dark grey | Fold / coordinate reusables | `FoldProperty`, `FoldManager`, `DebugHandle`, `SourceLoc`, `ProvenanceTransform` |
| Blue | Bundle / DSC structure | `SuperDsc`, `WrappedDesignSpaceConfig`, `DesignSpaceConfig` |
| Violet | Tensor objects | `LabeledDataStructure`, `MemoryOrganization`, `PrimaryDsInfo`, `ConstantInfo` |
| Green | Scheduling & coordinate objects | `ScheduleTreeNode`, `CoordinateContainer`, `CoordinateInfo`, `DataStageParam` |
| Amber | Compute objects | `ComputeOperation`, `attributes_` |

| Component | Role | Reference |
|---|---|---|
| **Object Hierarchy** | Full object tree of the entire JSON format | [JSON-object-Hierarchy.md](JSON-object-Hierarchy.md) |
| **SuperDsc** | Root object — holds fold properties, work-slice maps, core schedule, and the `dscs_[]` array | [superdsc-object.md](superdsc-object.md) |
| **DesignSpaceConfig** | Per-operation configuration — tensors, staging, schedule, and compute | [designspaceconfig.md](designspaceconfig.md) |
| **LabeledDataStructure** | Tensor descriptor — role, format, scale, and memory residency | [labeleddatastructure.md](labeleddatastructure.md) |
| **PrimaryDsInfo** | Tensor-type layout — memory dimension order and stick configuration | [primarydsinfo.md](primarydsinfo.md) |
| **DataStageParam** | Per-core tile sizes for steady-state and epilogue passes | [datastageparam.md](datastageparam.md) |
| **ScheduleTreeNode** | Memory allocation node — component, start addresses, coordinates | [scheduletreenode.md](scheduletreenode.md) |
| **ComputeOperation** | Compute specification — execution unit, op name, tensor references | [computeoperation.md](computeoperation.md) |
| **MemoryOrganization** | HBM / LX scratchpad residency flags per tensor | [memoryorganization.md](memoryorganization.md) |

---

| [← Previous: MLIR Complete Example](MLIR-complete-example.md) | [↑ Table of Contents](README.md) | [Next: Object Hierarchy →](JSON-object-Hierarchy.md) |
|:--|:--:|--:|
