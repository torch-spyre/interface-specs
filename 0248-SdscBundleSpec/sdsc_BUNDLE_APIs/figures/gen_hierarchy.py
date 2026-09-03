"""
Generate sdsc_json_hierarchy.png — visual object hierarchy of an SDSC JSON file.
Run from any directory; output is written next to this script.
"""

import os
import graphviz

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette ─────────────────────────────────────────────────────────────
C_ROOT   = "#1f2328"   # near-black  – root / entry point
C_SUPER  = "#2563a8"   # deep blue   – SuperDsc level
C_DSC    = "#3b82d4"   # mid blue    – DesignSpaceConfig level
C_TENSOR = "#7c5cd8"   # violet      – tensor / data objects
C_SCHED  = "#2d7a56"   # green       – scheduling / coordinate objects
C_COMP   = "#c0540a"   # amber       – compute objects
C_FOLD   = "#57606a"   # grey        – reusable FoldManager / FoldProperty
C_WHITE  = "#ffffff"
C_LIGHT  = "#f7f8fa"

def node(g, name, label, fillcolor, fontcolor=C_WHITE, shape="box", style="filled,rounded"):
    g.node(name, label=label, shape=shape, style=style,
           fillcolor=fillcolor, fontcolor=fontcolor,
           fontname="Helvetica", fontsize="11")

def edge(g, src, dst, label=""):
    g.edge(src, dst, label=label, fontsize="9", fontname="Helvetica",
           color="#57606a", arrowsize="0.7")

# ── build graph ────────────────────────────────────────────────────────────────
g = graphviz.Digraph(
    "sdsc_hierarchy",
    format="png",
    graph_attr={
        "rankdir": "TB",
        "splines": "ortho",
        "nodesep": "0.45",
        "ranksep": "0.7",
        "bgcolor": C_WHITE,
        "fontname": "Helvetica",
        "label": 'SDSC JSON File — Object Hierarchy',
        "labelloc": "t",
        "fontsize": "16",
        "fontcolor": C_ROOT,
        "pad": "0.4",
    },
    node_attr={"margin": "0.15,0.1"},
)

# ── Level 0: Root ──────────────────────────────────────────────────────────────
node(g, "ROOT",  "Root Object\n(operation key, e.g. \"matmul\")", C_ROOT)

# ── Level 1: SuperDsc ──────────────────────────────────────────────────────────
node(g, "SD",    "SuperDsc", C_SUPER)
edge(g, "ROOT", "SD")

# SuperDsc children
node(g, "FP",    "sdscFoldProps_\nArray<FoldProperty>",          C_FOLD,   C_WHITE)
node(g, "FM",    "sdscFolds_\nFoldManager",                      C_FOLD,   C_WHITE)
node(g, "CFP",   "coreFoldProp_\nFoldProperty",                  C_FOLD,   C_WHITE)
node(g, "DH",    "debug_handle_\nDebugHandle | null",            C_FOLD,   C_WHITE)
node(g, "META",  "numCoresUsed_\ncoreIdToDsc_\ncoreIdToWkSlice_\n…",  C_SUPER, C_WHITE)
node(g, "DSCS",  "dscs_[]\nArray<WrappedDesignSpaceConfig>",     C_SUPER)

for ch in ("FP", "FM", "CFP", "DH", "META", "DSCS"):
    edge(g, "SD", ch)

# FoldProperty detail
node(g, "FPROP", "FoldProperty\n─ factor_: integer\n─ label_: string", C_FOLD, C_WHITE)
edge(g, "FP", "FPROP")

# FoldManager detail (shared — shown once, referenced multiple times via label)
node(g, "FMDET", "FoldManager\n─ dim_prop_func[]: Const|Map|Affine|WkSplit\n─ dim_prop_attr[]: FoldProperty[]\n─ data_: Object",
     C_FOLD, C_WHITE)
edge(g, "FM", "FMDET")

# DebugHandle detail
node(g, "DHDET", "DebugHandle\n─ id: string\n─ aten_op: string | null\n─ ir_chain: string[]\n─ fused_from: DebugHandle[] ↺",
     C_FOLD, C_WHITE)
node(g, "SL",    "SourceLoc\n─ file, start_line\n─ start_col, end_line\n─ end_col",  C_FOLD, C_WHITE)
node(g, "PT",    "ProvenanceTransform\n─ kind: rewrite|fusion|\n  decomposition|clone|remap\n─ pass_name, reason",
     C_FOLD, C_WHITE)
edge(g, "DH",    "DHDET")
edge(g, "DHDET", "SL",   "source")
edge(g, "DHDET", "PT",   "transform_history[]")

# ── Level 2: WrappedDSC → DesignSpaceConfig ───────────────────────────────────
node(g, "WDSC",   "WrappedDesignSpaceConfig\n(operation name wrapper)", C_DSC)
node(g, "CONFIG", "DesignSpaceConfig",                                   C_DSC)
edge(g, "DSCS",   "WDSC")
edge(g, "WDSC",   "CONFIG")

# ── DesignSpaceConfig children ────────────────────────────────────────────────
node(g, "N_",    "N_: DataStructDims\n─ name_: string\n─ <dim>_: number",       C_DSC, C_WHITE)
node(g, "DIM2S", "dimToSymbolMapping_\ncoordinateMasking_\npdsRelation_\n…",     C_DSC, C_WHITE)
node(g, "DSP",   "dataStageParam_\nObject<coreId → Object>\n─ name_: core|corelet|row\n─ ss_: DataStructDims\n─ el_: DataStructDims",
     C_SCHED, C_WHITE)
node(g, "PDS",   "primaryDsInfo_\nObject<label → Object>\n─ layoutDimOrder_[]\n─ stickDimOrder_[]\n─ stickSize_[], stickRepl_[]",
     C_TENSOR, C_WHITE)
node(g, "ST",    "scheduleTree_[]\nArray<ScheduleTreeNode>",                     C_SCHED)
node(g, "LDS",   "labeledDs_[]\nArray<LabeledDataStructure>",                    C_TENSOR)
node(g, "CI",    "constantInfo_\nObject<index → ConstantInfo>",                  C_TENSOR, C_WHITE)
node(g, "COP",   "computeOp_[]\nArray<ComputeOperation>",                        C_COMP)

for ch in ("N_", "DIM2S", "DSP", "PDS", "ST", "LDS", "CI", "COP"):
    edge(g, "CONFIG", ch)

# ── ScheduleTreeNode ──────────────────────────────────────────────────────────
node(g, "STN",   "ScheduleTreeNode\n─ nodeType_: \"allocate\"\n─ name_: string\n─ component_: hbm|lx\n─ layoutDimOrder_[]\n─ maxDimSizes_[]\n─ isStartAddrSymbolic_\n─ indirectAllocType_",
     C_SCHED, C_WHITE)
node(g, "COORD", "coordinates_\nCoordinateContainer",                            C_SCHED, C_WHITE)
node(g, "CINFO_MAP", "coordInfo\nObject<dim → CoordinateInfo>",                  C_SCHED, C_WHITE)
node(g, "CINFO", "CoordinateInfo\n─ spatial: integer\n─ temporal: integer\n─ elemArr: integer\n─ padding: nopad | lowered_padded\n  | padded_nozeropad | padded_wzeropad\n  | padded_fullspan | padded_fullspan_wunneeded\n─ folds: FoldManager",
     C_SCHED, C_WHITE)

edge(g, "ST",       "STN")
edge(g, "STN",      "COORD",     "coordinates_")
edge(g, "COORD",    "CINFO_MAP")
edge(g, "CINFO_MAP","CINFO")

# startAddressCoreCorelet_ → FoldManager (reuse label)
node(g, "ADDR_FM", "startAddressCoreCorelet_\nFoldManager", C_FOLD, C_WHITE)
edge(g, "STN", "ADDR_FM")

# ── LabeledDataStructure ──────────────────────────────────────────────────────
node(g, "LDS1",  "LabeledDataStructure\n─ ldsIdx_: integer\n─ dsName_: string\n─ dsType_: INPUT|OUTPUT\n  |KERNEL|KERNEL_IDX\n─ dataFormat_: FP16|FP32|BF16…\n─ wordLength: number\n─ scale_: number[]",
     C_TENSOR, C_WHITE)
node(g, "MO",    "memOrg_: MemoryOrganization",                                  C_TENSOR, C_WHITE)
node(g, "HBM",   "hbm: MemorySlot\n─ isPresent: 0|1\n─ isPadded: 0|1\n─ isZeroPadded: 0|1\n─ dsOffset: integer\n─ allocateNode_: string",
     C_TENSOR, C_WHITE)
node(g, "LXS",   "lx: MemorySlot\n─ isPresent: 0|1\n─ isPadded: 0|1\n─ isZeroPadded: 0|1\n─ dsOffset: integer\n─ allocateNode_: string",
     C_TENSOR, C_WHITE)

edge(g, "LDS",  "LDS1")
edge(g, "LDS1", "MO")
edge(g, "MO",   "HBM")
edge(g, "MO",   "LXS")

# ── ConstantInfo ──────────────────────────────────────────────────────────────
node(g, "CINFO2", "ConstantInfo\n─ name_: string\n─ dataFormat_: string\n─ data_: FoldManager",
     C_TENSOR, C_WHITE)
edge(g, "CI", "CINFO2")

# ── ComputeOperation ─────────────────────────────────────────────────────────
node(g, "COMP",  "ComputeOperation\n─ exUnit: sfp | pt\n─ opFuncName: string\n─ location: \"Inner\"\n─ inputLabeledDs: string[]\n─ outputLabeledDs: string[]\n─ interimLabeledDs: string[]\n─ indirectAccessIndexLabeledDs: string[]",
     C_COMP, C_WHITE)
node(g, "ATTR",  "attributes_\n─ dataFormat_: string\n─ fidelity_: regular|fast",
     C_COMP, C_WHITE)

edge(g, "COP",  "COMP")
edge(g, "COMP", "ATTR")

# ── Render ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "sdsc_json_hierarchy")
g.render(out_path, cleanup=True)
print(f"Written: {out_path}.png")
