# LX Relayout SHUFFLE Contract

Authors: Adnan Hoque
Status: Draft extension to the SuperDSC Bundle specification

## Summary

This document defines the SuperDSC/DLDSC contract for an explicit LX-to-LX relayout:

```text
S1 in LX -> SHUFFLE -> S2 in LX
```

S1 and S2 are separate frontend-allocated tensor states. Their DLDSC coordinates define the logical copy, permutation, or replication. The backend validates and lowers supported affine mappings while retaining freedom over the physical ring and local-copy schedule.

This document extends the [SuperDSC Bundle specification](./SuperDSC-Bundle.md). The compiler-policy rationale belongs in the corresponding [Torch-Spyre LX relayout RFC](https://github.com/torch-spyre/torch-spyre/issues/3224).

## Scope

The v1 contract covers bounded, static, affine, non-arithmetic relayouts where:

- the source and destination are both in LX;
- the complete S1 and S2 allocations are known before backend lowering;
- every destination element has one logical source element;
- one source element may feed multiple destinations;
- destination writes do not overlap; and
- SHUFFLE completes before the dependent consumer.

This is sufficient to represent one-to-one all-to-all shuffle and grouped all-gather. It does not define arithmetic reductions, dynamic routing, or streamed/partially resident destinations.

## Normative terms

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as requirements on producers and consumers of the SuperDSC interface.

## Definitions

| Term | Definition |
|---|---|
| S1 | Pre-relayout tensor state. Its allocation coordinates describe where producer-owned values reside. |
| S2 | Post-relayout tensor state. Its allocation coordinates describe where consumer-required values must reside. |
| SHUFFLE | A copy-only DLDSC operation that materializes S2 from S1. |
| source domain | The logical tensor coordinates present in S1. |
| destination domain | The logical tensor coordinates required in S2. |
| representable mapping | A mapping expressible by the affine coordinate and fold model in this specification. |
| supported mapping | A representable mapping accepted by a particular backend version and target. |

Collective names are descriptive, not additional opcodes:

- one-to-one partitioned S1 to partitioned S2 is an all-to-all shuffle;
- partitioned S1 to repeated S2 within a group is an all-gather; and
- both are encoded as `SHUFFLE` with different coordinates.

## Frontend obligations

For every emitted SHUFFLE, the frontend MUST:

1. allocate S1 and S2 explicitly in LX;
2. provide the source and destination base address for each participating core/corelet;
3. provide the logical layout and coordinate folds for both tensor states;
4. provide `coreIdToWkSlice_` for both tensor states;
5. ensure S1 and S2 are disjoint for the duration of the transfer;
6. ensure S1 remains live until SHUFFLE reads it;
7. ensure S2 remains live from SHUFFLE through its consumer;
8. schedule SHUFFLE after the producer and before the consumer;
9. emit complete, non-overlapping destination coverage;
10. avoid SHUFFLE for partial reductions or arithmetic combination; and
11. retain an HBM fallback when the mapping is unsupported or the allocations do not fit.

The frontend SHOULD avoid emitting a mapping outside the backend-supported affine subset. Capability discovery/versioning is not defined in v1.

## Backend obligations

For a supported SHUFFLE, the backend MUST:

1. interpret S1 and S2 as independent tensor states;
2. derive the logical movement from their coordinate mappings;
3. materialize the complete S2 value required by the descriptor;
4. respect the producer -> SHUFFLE -> consumer dependencies;
5. write only within the explicit S2 allocation;
6. preserve tensor values exactly, with no arithmetic combination; and
7. reject an unsupported or malformed mapping rather than silently changing semantics.

The backend MAY choose any valid physical realization, including different ring routes, transfer decompositions, local copies, packet sizes, and synchronization strategies. Those choices are not part of this interface.

In the v1 full-resident model, the backend MUST NOT assume an additional hidden full-size S2 allocation. Bounded private workspace for implementation details is permitted only within the target's documented backend reservation.

## Descriptor shape

A SHUFFLE is represented by one DLDSC with two allocated tensors:

| DLDSC element | Requirement |
|---|---|
| `opFuncName` | `shuffle` / `SHUFFLE` according to the existing case convention |
| `scheduleTree_[0]` | S1 input allocation in LX (`ldsIdx_ = 0`) |
| `scheduleTree_[1]` | S2 output allocation in LX (`ldsIdx_ = 1`) |
| `component_` | `lx` for both allocations |
| `layoutDimOrder_` | Physical dimension order of the corresponding state |
| `startAddressCoreCorelet_` | Per-core/corelet LX base address |
| `coordinates_.coreIdToWkSlice_` | Per-core logical slice ownership for the corresponding state |
| `coordinates_.coordInfo` | Affine fold definition for each semantic dimension |
| `coreIdToDscSchedule` | Schedule placement for participating cores |

The two allocation rows are authoritative. A consumer compute split alone is not sufficient to define S1; the S1 allocation row records the producer distribution that remains live at the boundary.

### Abbreviated form

The following is illustrative. Fields unrelated to this contract are omitted.

```json
{
  "6_shuffle": {
    "dscs_": [
      {
        "shuffle": {
          "scheduleTree_": [
            {
              "name_": "allocate-Tensor0_lx",
              "ldsIdx_": 0,
              "component_": "lx",
              "layoutDimOrder_": ["out", "in", "x"],
              "startAddressCoreCorelet_": "<S1 bases>",
              "coordinates_": {
                "coreIdToWkSlice_": "<producer distribution>",
                "coordInfo": "<S1 affine folds>"
              }
            },
            {
              "name_": "allocate-Tensor1_lx",
              "ldsIdx_": 1,
              "component_": "lx",
              "layoutDimOrder_": ["out", "in", "x"],
              "startAddressCoreCorelet_": "<S2 bases>",
              "coordinates_": {
                "coreIdToWkSlice_": "<consumer distribution>",
                "coordInfo": "<S2 affine folds>"
              }
            }
          ]
        }
      }
    ],
    "coreIdToDscSchedule": "<producer-before-shuffle-before-consumer>"
  }
}
```

## Coordinate invariants

### Static affine mapping

Every semantic dimension participating in SHUFFLE MUST have a static affine fold. A fold used for addressing MUST describe the same cardinality and stride as the physical allocation.

### Complete destination coverage

The union of destination coordinates MUST cover every logical element read by the consumer. No two logical source elements may write different values to the same destination element.

### Replication

Replication is expressed by mapping one logical source coordinate to multiple destination core coordinates. Repeated destination ownership is legal when each repeated destination receives the same logical value.

No `collective=all_gather` classification field is required for semantics. The all-gather is visible in the producer and destination coordinate geometry.

### Corelet folds

Corelet cardinality is part of the physical addressing contract and MUST be represented on the semantic dimension that it partitions.

For FP16 `D=128` represented by two 64-element corelets, the `in` coordinate fold contains:

```json
{
  "dim_prop_func": [
    {"Affine": {"alpha_": 128, "beta_": 0}},
    {"Affine": {"alpha_": 64, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 1, "label_": "core_fold"},
    {"factor_": 2, "label_": "corelet_fold"},
    {"factor_": 1, "label_": "row_fold"},
    {"factor_": 128, "label_": "elem_arr_0"}
  ]
}
```

Omitting or attaching the two-way corelet fold to a different semantic dimension changes the address geometry and MUST be rejected.

### Allocation and lifetime

S1 and S2 MAY reuse the same numeric address at non-overlapping times, but MUST NOT alias at the SHUFFLE boundary. For the full-resident v1 path, both allocations are simultaneously live during the transfer.

## Attention all-gather example

Consider:

```text
K_scaled: [1, 4, 4096, 128]
producer split: {H: 4, Lk: 8}
consumer split: {H: 4, Lq: 8}
```

For each head, cores are arranged in an eight-core group.

S1 distribution:

```text
core 0: K[head,   0: 512, :]
core 1: K[head, 512:1024, :]
...
core 7: K[head,3584:4096, :]
```

S2 distribution:

```text
core 0: K[head, 0:4096, :]
core 1: K[head, 0:4096, :]
...
core 7: K[head, 0:4096, :]
```

The same pattern repeats for the remaining three head groups. The destination `coreIdToWkSlice_` repeats the same full-K logical view across the eight destinations, while the S1 mapping assigns one distinct K shard to each source.

For FP16:

```text
S1 per core = 512 * 128 * 2 bytes = 128 KiB
S2 per core = 4096 * 128 * 2 bytes = 1 MiB
```

The frontend allocates both. The backend copies each source shard into its corresponding offset in every S2 within the group. The exact transfer schedule is backend-internal.

## Interaction with restickify

SHUFFLE changes tensor distribution. Restickify changes local tensor layout.

If the source and consumer use incompatible stick layouts, the frontend emits the existing restickify operation before SHUFFLE:

```text
producer state -> restickify -> S1 -> SHUFFLE -> S2 -> consumer
```

The restickify operation MAY use LX allocations when legal. Its historical op name does not change the allocation contract: `component_ = lx` and the allocation addresses determine residency.

The v1 SHUFFLE contract assumes that S1 and S2 have copy-compatible local layouts. A backend MAY fuse or reorder legal local movement internally, but observable S2 semantics remain unchanged.

## Capability model

The coordinate model is more expressive than any one backend implementation. Therefore, representability and backend support are distinct:

| Geometry | Representable by DLDSC | Required v1 backend support |
|---|---:|---:|
| Uniform one-to-one all-to-all shuffle | yes | yes |
| Uniform grouped all-gather | yes | yes |
| Local copy on the same core | yes | yes |
| Non-uniform affine fanout | possibly | no |
| Multiple independent consumer views sharing S2 | yes | no |
| Dynamic/data-dependent routing | no | no |
| Arithmetic reduction | no | no |
| Streamed partial S2 | not defined by this document | no |

Backend implementation details are deliberately opaque across this interface. Backend support SHOULD, however, be versioned or discoverable so the frontend can make a deterministic fallback decision before code generation.

## Validation

A conforming producer/backend pair SHOULD test at least:

1. one-to-one shuffle with patterned values;
2. grouped all-gather with patterned source shards;
3. same-core local copies mixed with cross-core copies;
4. a two-corelet `D=128` operand;
5. non-zero, distinct S1 and S2 LX base addresses;
6. malformed corelet cardinality rejection;
7. incomplete destination coverage rejection;
8. overlapping destination-write rejection; and
9. capacity failure with frontend HBM fallback.

Structural validation MUST confirm:

- both allocations are LX;
- S1 and S2 addresses are disjoint while live;
- SHUFFLE precedes the consumer;
- the targeted HBM handoff is absent; and
- the consumer reads S2.

## Failure behavior

The frontend MUST NOT emit SHUFFLE when its static checks fail or S2 cannot be allocated. The backend MUST reject unsupported valid geometry and malformed geometry with a diagnosable error.

The interface does not define a backend-initiated HBM fallback for an already emitted explicit SHUFFLE. Fallback selection occurs before the final descriptor bundle is emitted.

## Compatibility and versioning

This contract reuses existing SuperDSC/DLDSC fields and the existing SHUFFLE op. No communication-class metadata is required for correctness.

The following are future-compatible extensions, not v1 behavior:

- a capability/version field for supported affine subsets;
- staged or streamed destination residency;
- explicit cost hints;
- overlap/synchronization hints; and
- arithmetic collective operations.

## References

- [SuperDSC Bundle specification](./SuperDSC-Bundle.md)
- [Torch-Spyre LX relayout RFC](https://github.com/torch-spyre/torch-spyre/issues/3224)
- [Torch-Spyre tracking epic](https://github.com/torch-spyre/torch-spyre/issues/3049)
- [Torch-Spyre implementation](https://github.com/torch-spyre/torch-spyre/pull/2939)
- Deeptools implementation: [ai-chip-toolchain/deeptools#4408](https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4408)
