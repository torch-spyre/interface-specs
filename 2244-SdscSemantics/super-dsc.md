# SuperDSC Semantics — Documentation of SuperDSC Fields
### - Companion to Section [sdsc.json filling](https://github.com/vswagath1989/torch-spyre/blob/sdsc-bundle-spec/RFCs/0248-SdscBundleSpec/SuperDSC-Bundle.md#sdscjson-filling) of [SuperDSC Bundle Interface Specification](https://github.com/vswagath1989/torch-spyre/blob/sdsc-bundle-spec/RFCs/0248-SdscBundleSpec/SuperDSC-Bundle.md))

This document is intended to complement the SuperDSC Bundle Spec by providing any needed background / details on SDSC’s fields and guidance on how to fill them beyond what can be covered in the Spec. The intended audience is teams coding up the SDSC specification from the frontend of Torch Spyre stack for consumption/further expansion by the DeepTools backend. Scope is limited to the fields that need to be filled from Torch-Spyre frontend. It is not intended to serve as a fully-contained standalone document, but has to be read along with one or more of the Spec itself, actual sdsc.json files, and relevant DeepTools source code. A claude-generated SuperDSC class hierarchy can be found [here](https://ibm.ent.box.com/file/2168779496917).

This is a working document that is expected to evolve over time (in a crowd-sourced manner within IBM) both in content to capture a good collection of nuanced and non-trivial aspects of SDSC as well as format and organization. Clarifications sought can be added to the [Questions Section](#dscs_idxcomputeop_-class-computeopinfo).

SuperDSC is a json representation of an operation to be performed by the Spyre backend, DeepTools. It also includes specifications of the input and output tensors needed by the operation.

SuperDSC consists of a few top-level fields and an array of structures termed dscs_[] (design  space configs). Each dsc_ entry consists of some leaf (final) fields and a few composite (non-leaf) ones that can be drilled down into.

The usage of a field can be better understood by referring to sample sdsc files.


## Top-Level Fields

Class SuperDsc in @deeptools/dsc/superdsc.h has definitions of top-level fields. Not all fields of that class have been captured here, only the ones seen commonly in SDSC’s generated.

Not all fields need to be filled by the front-end. Which ones need to be filled is TBD.

The root key of the SDSC json is name\_ typically indicating the main operation performed.

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 20%" />
<col style="width: 27%" />
<col style="width: 23%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>S.No.</strong></th>
<th style="text-align: center;"><strong>Field Name (Mandatory/Optional Indicator)</strong></th>
<th style="text-align: center;"><strong>Purpose / Functionality</strong></th>
<th style="text-align: center;"><strong>How to fill</strong></th>
<th style="text-align: center;"><strong>Additional Comments/Questions</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5"><strong>Core Metadata</strong></td>
</tr>
<tr>
<td><ol type="1">
<li></li>
</ol></td>
<td>name_</td>
<td>Identifier for the sdsc.</td>
<td>Used as root key for the rest of the json. <mark>E.g., { “bmm” : { rest of sdsc blocks and fields.} }</mark> Here bmm is the name.</td>
<td></td>
</tr>
<tr>
<td><ol start="2" type="1">
<li></li>
</ol></td>
<td>numCoresUsed_</td>
<td>Total no. of cores used</td>
<td></td>
<td></td>
</tr>
<tr>
<td colspan="5"><strong>Folding Information</strong></td>
</tr>
<tr>
<td><ol start="3" type="1">
<li></li>
</ol></td>
<td><strong>sdscFoldProps_</strong></td>
<td><p>Vector of FoldDimProp containing fields <b>factor_</b> and <b>label_</b>. Most sdsc’s have this filled as "sdscFoldProps_" : [</p>
<p>{"factor_" : 1, "label_" : "time"}</p>
<p>]</p>
<p>Definition can be found in class <strong>FoldDimProp</strong> in @dsc/foldmanager/foldInfrastructure.h</p></td>
<td>These denote folds over time. The default values are good from front end.</td>
<td></td>
</tr>
<tr>
<td><ol start="4" type="1">
<li></li>
</ol></td>
<td><strong>sdscFolds_</strong></td>
<td>Captures details of <strong>FoldManager</strong> structure containing <strong>dim_prop_func</strong>, <strong>dim_prop_attr</strong>, and <strong>data_</strong> fields.</td>
<td>Defaults good from front end.</td>
<td></td>
</tr>
<tr>
<td><ol start="5" type="1">
<li></li>
</ol></td>
<td>coreFoldProp_</td>
<td>Contains subfields <strong>factor</strong> and <strong>label</strong>.</td>
<td><p>From SDSC Spec:</p>
<p>factor=maxCoreId, label=’core’</p></td>
<td></td>
</tr>
<tr>
<td><ol start="6" type="1">
<li></li>
</ol></td>
<td>coreletFoldProp_</td>
<td>--- same as above --</td>
<td>Factor=2, label=’corelet’</td>
<td>Most sdsc’s have corelet factor specified as 1. Should it be 1 or 2?</td>
</tr>
<tr>
<td><ol start="7" type="1">
<li></li>
</ol></td>
<td>folded_sdsc_name_</td>
<td></td>
<td></td>
<td>Does the name here have to match the name_ of the whole sdsc?</td>
</tr>
<tr>
<td><ol start="8" type="1">
<li></li>
</ol></td>
<td>fold_coord_</td>
<td><p>std::deque&lt;int64_t&gt; fold_coord_;</p>
<p><strong><mark>From claude:</mark></strong></p>
<p>For a folded SDSC with multiple fold types, <strong>fold_coord_</strong> stores the index at each fold dimension. Specifies which <strong>unfolded variant</strong> an SDSC represents by storing its <strong>coordinate indices in folded space</strong>.</p></td>
<td></td>
<td>This field seems to be populated only for unfolded variants of the SDSC.</td>
</tr>
<tr>
<td colspan="5"><strong>Dimensions and Work Slicing</strong></td>
</tr>
<tr>
<td><ol start="9" type="1">
<li></li>
</ol></td>
<td>N_</td>
<td>List of all dimensions across tensors used by ops in all dsc_’s in the sdsc. Contains each dimension’s size across all cores. For dimensions that are padded (such as convolution’s image dimensions) padding details are also included.</td>
<td>Permissible fields of this entry can be found in DataStructDim in dsc/dims.h. Negative value indicates that a dimension is not used. Inline comments explain the purpose of non-intuitive fields such as coreletSplit_ and rowSplit_.</td>
<td></td>
</tr>
<tr>
<td><ol start="10" type="1">
<li></li>
</ol></td>
<td>unPadN_</td>
<td>Probably description of unpadded version of dimensions in N_</td>
<td></td>
<td>Should this have different values than N_ when there is padding?</td>
</tr>
<tr>
<td><ol start="11" type="1">
<li></li>
</ol></td>
<td>numWkSlicesPerDim_</td>
<td>A map keyed by the dimension name indicating the number of slices into which each dimension is split. The product of the slices over all dimensions should equal the number of cores (across which the operation is executed).</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="12" type="1">
<li></li>
</ol></td>
<td>coreIdToWkSlice_</td>
<td>Map from core id to the slice of a dimension assigned to it. Nested map. Outer key is core id. Inner key is dimension name. The slice number assigned to a core ranges from 0 to the total slice count for the dim indicated by numWkSlicesPerDim_. See example sdsc's.</td>
<td></td>
<td></td>
</tr>
<tr>
<td colspan="5"><strong>Data Structures</strong></td>
</tr>
<tr>
<td><ol start="13" type="1">
<li></li>
</ol></td>
<td>dscs_</td>
<td> Array of <strong>DesignSpaceConfig</strong> (or DataStageConfig) structures. Vector to express core work mapping for an operation. With balanced work division, only one entry in the vector is needed.</td>
<td></td>
<td>Will need examples when work is not balanced.</td>
</tr>
<tr>
<td colspan="5"><strong>Maps</strong></td>
</tr>
<tr>
<td><ol start="14" type="1">
<li></li>
</ol></td>
<td>coreIdToDsc_</td>
<td>Mapping from core id to dsc number when the sdsc contains multiple dsc’s.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="15" type="1">
<li></li>
</ol></td>
<td><strong>ldsShareInfo</strong>_</td>
<td>LabeledDS sharing information. Tracks tensor sharing across multiple DSC instances.</td>
<td></td>
<td><p>Claude-generated details to be reviewed in <a href="LDSSHAREINFO.md">LDSSHAREINFO.md</a>.</p>
<p>Since labeledDS is part of a dsc_ element, how will the labeledDS be identified?</p></td>
</tr>
<tr>
<td><ol start="16" type="1">
<li></li>
</ol></td>
<td>opFuncsUsed_</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="17" type="1">
<li></li>
</ol></td>
<td>prodConsList</td>
<td></td>
<td></td>
<td>Should this be filled by the frontend?</td>
</tr>
<tr>
<td colspan="5"><strong>Scheduling related</strong></td>
</tr>
<tr>
<td><ol start="18" type="1">
<li></li>
</ol></td>
<td><strong>coreIdToDscSchedule</strong></td>
<td>Vector of <strong>DscScheduleStep</strong> schedule steps specified for each core. Defines the execution sequence of operations on each core by specifying which data and dl dsc operations execute on this core, the order they execute in, and data-flow dependencies.</td>
<td></td>
<td>Details in <a href="COREIDTODSCSCHEDULE.md">COREIDTODSCSCHEDULE.md</a></td>
</tr>
<tr>
<td><ol start="19" type="1">
<li></li>
</ol></td>
<td>target_</td>
<td><p>Can be one of SENTIENT,  SENULATOR,</p>
<p>  SENPCFG,  SENTF,  SYSTEMC,  R5SS, HOST. Specifies the target execution function backend/hardware platform for the SDSC.</p></td>
<td></td>
<td>Should the frontend always set to SENULATOR?</td>
</tr>
<tr>
<td colspan="5">DataDSC related</td>
</tr>
<tr>
<td><ol start="20" type="1">
<li></li>
</ol></td>
<td>dataOpdscs_</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

## Fields of dscs\_:

dscs\_ is an array of structures whose definition in sdsc.json closely mirrors the definition of class DesignSpaceConfig in designspaceconfig.h. dscs\_ array is a part of the top-level sdsc\_ block.

Several fields are described in the SuperDSC bundle specification document. Non-trivial fields not fully described in the specification are provided below. As with the sdsc above, each entry of the array has a name that encloses the fields in the table below. E.g. \[{“bmm”: {}}\]

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 20%" />
<col style="width: 25%" />
<col style="width: 24%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>S.No.</strong></th>
<th style="text-align: center;"><strong>Field Name (Mandatory/Optional Indicator)</strong></th>
<th style="text-align: center;"><strong>Purpose / Functionality</strong></th>
<th style="text-align: center;"><strong>How to fill</strong></th>
<th style="text-align: center;"><strong>Additional Comments/Questions</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><ol type="1">
<li></li>
</ol></td>
<td style="text-align: center;">N_</td>
<td>Similar to sdsc.N_. Details of dimensions used by a specific dsc_.</td>
<td>Padding details provided in sub-field <strong>paddingSizes</strong>_ of type DimPaddingSizes (in @dsc/dims.h). One entry per dim that is padded.</td>
<td>See <a href=#padding>Sec. Padding</a> for more details.</td>
</tr>
<tr>
<td><ol start="2" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">dscN_</td>
<td>Also of type dataStructDim_ (like N_)</td>
<td>What is the purpose? How is it different from N_.</td>
<td>Is this required?</td>
</tr>
<tr>
<td><ol start="3" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">coreIdsUsed_</td>
<td>List of cores used by the ops in this dsc</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="4" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>dataStageParam</strong>_</td>
<td><p>Vector of DataStage structs.</p>
<p>std::map&lt;int, dsc2::DataStage&gt; dataStageParam_;</p>
<p>Sizes per dimension for a single core.</p>
<p>Defined in @dsc/dsc2.h.</p>
<p>Each element in the vector requires fields ss_ and el_, each of type dataStructDim_ (like N_).</p>
<p>Specifies sizes per dimension for each core.The overall field is keyed via "0".</p></td>
<td><p><strong>ss</strong> stands for steady_state and <strong>el</strong> for epilogue. Related to work division across cores. el will likely be different from ss only when the work division across cores is not balanced.</p>
<p>From SDSC spec:</p>
<p>add one entry with key 0, and fill ss_ and el_ with same data (name should be “core”)</p></td>
<td>
   <p>Like N_, padding details for window/padded operations are added to padding sun-fields within dataStageParam_. See See <a href=#padding>Sec. Padding</a> for more details. </p>
   <p/>
   <p>When should ss_ and el_ be different?</p>
   </td>
</tr>
<tr>
<td><ol start="5" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">labeledDs_</td>
<td>Vector of LabeledDsInfo struct. Each entry represents a physical tensor used in the dsc_’s operation. Both input and output tensors need to be listed. Fields of <strong>LabeledDsInfo</strong> to populate can be found in @dsc/dscdefn.h. Explained further below in <a href="#fields-of-labeledds_-vector-of-labeleddsinfo">labeledDs section.</a></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="6" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">primaryDsInfo_</td>
<td><p>Defines DsType’s used in the dsc. A DsType denotes a tensor type and defines its dimensions, also specifying which among the dimensions is the stick dimension.</p>
<p>Currently defined types are <strong>enum DsTypes { INPUT, OUTPUT, KERNEL, KERNEL_IDX, NOT_SET };</strong></p></td>
<td><p> Provides a mapping from the DsType name to PrimaryDsInfo, which captures the details of the DsType as in:
std::map<DsTypes, PrimaryDsInfo> primaryDsInfo_ </p>
<p>Contains one entry for each DsType used in the dsc. </p>
<p>A DsType corresponds to a list of dimensions and stick dimension used by a tensor. Multiple physical tensors can share a DsType.</p></td>
<td><p>struct PrimaryDsInfo {<br>
  std::vector&lt;PrimaryDimTypes&gt; layoutDimOrder_;<br>
  std::vector&lt;PrimaryDimTypes&gt; stickDimOrder_;<br>
  std::vector&lt;double&gt; stickSize_;<br>
  std::vector&lt;int&gt; stickRepl_;<br>
};</p></td>
</tr>
<tr>
<td><ol start="7" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">pdsRelation_</td>
<td>Has a Boolean isPdsReuse sub-field</td>
<td></td>
<td>When should this be set?</td>
</tr>
<tr>
<td><ol start="8" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">ChipD_, ChipletD_, Cored_, CoreletD_</td>
<td>All of type DataStructDims (like N_)</td>
<td></td>
<td>When should these be filled?</td>
</tr>
<tr>
<td><ol start="9" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">B_, T_, Tel_, P_, Pel_</td>
<td>All of type DataStructDims (like N_)</td>
<td></td>
<td>When should these be filled?</td>
</tr>
<tr>
<td><ol start="10" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><p>loopOrder_, loopProperties_</p>
<p>auxLoopOrder_</p></td>
<td>Should the front-end fill information related to loops?</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="11" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">dimToSymbolMapping_</td>
<td></td>
<td>See SDSC Bundle Spec</td>
<td><p>When are symbolic dimensions required?</p>
<p>Illustration with examples required.</p></td>
</tr>
<tr>
<td><ol start="12" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">constantInfo_</td>
<td></td>
<td>See SDSC Bundle Spec</td>
<td></td>
</tr>
<tr>
<td><ol start="13" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">scheduleTree_</td>
<td><p>Schedule of computations?</p>
<p>Sub-fields explained in <a href="#scheduletree_-class-scheduletree">ScheduleTree section</a> below.</p></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="14" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">computeOp_</td>
<td>Vector of ComputeOpInfo structures defined in @dsc/dscdefn.h. Describes the compute operations performed by the dsc_ of the SDSC. More than one op will be specified in case of fused operations.</td>
<td>Most fields are self explanatory. Rest covered below.</td>
<td></td>
</tr>
</tbody>
</table>

### Fields of labeledDs\_ (Vector of LabeledDsInfo): 

Each entry in labeledDs\_ corresponds to a physical tensor used by the dsc.

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 20%" />
<col style="width: 25%" />
<col style="width: 24%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>S.No.</strong></th>
<th style="text-align: center;"><strong>Field Name (Mandatory/Optional Indicator)</strong></th>
<th style="text-align: center;"><strong>Purpose / Functionality</strong></th>
<th style="text-align: center;"><strong>How to fill</strong></th>
<th style="text-align: center;"><strong>Additional Comments/Questions</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><ol type="1">
<li></li>
</ol></td>
<td style="text-align: center;">ldsIdx</td>
<td>Index number</td>
<td><strong>Is there any rule to be followed in assigning the index numbers?</strong></td>
<td></td>
</tr>
<tr>
<td><ol start="2" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">dsName</td>
<td>A distinct name for the tensor. Used to identify the input and output tensors associated with a computeOp_.</td>
<td></td>
<td><p>Are there other uses?</p>
<p>The name of the tensor used in computeOp_ seems to append idx&lt;ldsIdx&gt; to the dsName in labeledDs_. E.g., A labeledDS_ with <strong>"dsName_": "convolution-Tensor0"</strong> and <strong>"ldsIdx_": 0</strong> is denoted as <strong>"convolution-Tensor0-idx0".</strong> </p></td>
</tr>
<tr>
<td><ol start="3" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">dsType</td>
<td>dsType of this tensor. Should be a type defined in primaryDsInfo_ of the parent dsc_. Refer to primaryDsInfo_ field in <a href="#fields-of-dscs_">section above</a>.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="4" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">isStatic_, isFirstUse_, isExternal_</td>
<td></td>
<td>Should these be filled?</td>
<td></td>
</tr>
<tr>
<td><ol start="5" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>scale_</strong></td>
<td><p>std::vector&lt;double&gt; scale_</p>
<p>1 entry per dimension present in the dsType of this tensor. (As many entries as the number of dimensions in this tensor.)</p></td>
<td><p>From SDSC spec:</p>
<ul>
<li><p>1 is normal, -1 is reduced / broadcasted, -2 is reduced / broadcasted stick dimension.</p></li>
<li><p>order matches layoutDimOrder_ in primaryDsInfo</p></li>
</ul></td>
<td>In matmul (and batch matmul), each output element is obtained by reducing a (row, column) combination of the input tensors. Using the conventional names of x, mb, in, out for the dimensions, does this mean <strong>in</strong> dimension of the first and second tensors should have a scale of -1? Or because these don't vanish in the input tensors should they remain as 1? What should be the scale vector for the output tensor?</td>
</tr>
<tr>
<td><ol start="6" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">density_</td>
<td>Is this needed?</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="7" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">wordLength</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="8" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">dataFormat_</td>
<td>One of the formats defined in enum DataFormats in @util/sendefs.h. Indicates the hardware data format for this tensor.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="9" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>memOrg_</strong></td>
<td><p>Indicates memory residency (HBM vs LX).</p>
<p>std::map&lt;SenComponents, MemOrg&gt; memOrg_</p></td>
<td>MemOrg defined in @dsc/dscdefn.h contains fields isPresent, isPadded, isZeroPadded, zpadGapFront, gapPerDim, dsOffset, and allocateNode_</td>
<td><p>What is the difference between isPadded and isZeroPadded?</p>
<p>Typically only isPresent is set. Which other fields need to be filled and when?</p></td>
</tr>
<tr>
<td><ol start="10" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><p>dataTransfers_</p>
<p>hbmStartAddress_</p>
<p>lxStartAddress_</p>
<p>hbmSize_</p>
<p>lxSize_</p>
<p>lxBufferSize_</p>
<p>totSlicesPerDim_</p>
<p>coreStateInit_</p></td>
<td>Should any of these fields be filled by the front-end?</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

### scheduleTree\_ (class ScheduleTree):

\_scheduleTree is a tree (list) of ScheduleNodes which could be of types BLOCK,    LOOP,    TRANSFER,    COMPUTE,    SYNC,    CONDITION,    ALLOCATE,    STICKMASK, (among others).

Fields of ScheduleNode of type ALLOCATE. One allocate node needs to be added per tensor in LabeledDs. Only ALLOCATE nodes need to be filled from the front end.

<table style="width:100%;">
<colgroup>
<col style="width: 6%" />
<col style="width: 21%" />
<col style="width: 24%" />
<col style="width: 24%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>S.No.</strong></th>
<th style="text-align: center;"><strong>Field Name (Mandatory/Optional Indicator)</strong></th>
<th style="text-align: center;"><strong>Purpose / Functionality</strong></th>
<th style="text-align: center;"><strong>How to fill</strong></th>
<th style="text-align: center;"><strong>Additional Comments/Questions</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><ol type="1">
<li></li>
</ol></td>
<td style="text-align: center;">nodeType_</td>
<td>Should be set to <strong>allocate</strong>.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="2" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">name_</td>
<td>Any easy to identify name</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="3" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">ldsIdx_</td>
<td>Same as ldsIdx assigned in labeledDS?</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="4" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">component_</td>
<td>One of SenComponents such as HBM, LX to indicate memory residency</td>
<td></td>
<td>How to decide which mem component to use from the front end?</td>
</tr>
<tr>
<td><ol start="5" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>padding_</strong></td>
<td>Padding type of enum PadType for each dimension of the tensor that is padded.</td>
<td></td>
<td><p>enum class PadType {</p>
<p>  NOPAD,</p>
<p>  LOWERED_PADDED,    </p>
<p>  PADDED_NOZEROPAD,</p>
<p>  PADDED_WZEROPAD,  </p>
<p>  PADDED_FULLSPAN,  </p>
<p>  PADDED_FULLSPAN_WUNNEEDED,</p>
<p>};</p>
<p>See <a href=#padding>Sec. Padding</a> for some description of what the types indicate.</p></td>
</tr>
<tr>
<td><ol start="6" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">layoutDimOrder_</td>
<td>Order of the dimensions on the device.</td>
<td>The actual layout (beyond a stick) of the tensor is specified (from inner to outer). For example, in a 2D tensor with dimensions out=512, mb=4 elements and out=64 elements in the stick, the tensor will be [stick-out=64][layout-mb=4][layout-out=8]. So the layoutDimOrder_ should be [mb][out] and the size can be [-1, -1] i.e. all elements outside of stick. More details at <a href="#layout-dimorder-layoutdimorder_">Layout Dimorder</a>.</td>
<td>Is there a way to determine what device layouts to specify for different tensors for them to be compatible with what DeepTools expects.
E.g., specifying [j, i, mb, in] for both input and output tensors does not produce correct results by specifying [j, I, mb, in] for input and [mb, j, i, in] works.
</td>

</tr>
<tr>
<td><ol start="7" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">maxDimSizes_</td>
<td></td>
<td><p>From SDSC Spec:</p>
<p>fill maxDimSizes_ in AllocateNode of value tensor to set page size</p></td>
<td><p>Is setting to -1 always sufficient?</p>
<p>What is page size?</p></td>
</tr>
<tr>
<td><ol start="8" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>startAddressCoreCorelet</strong>_</td>
<td><p>Start address per core.</p>
<p>More details at <a href="#folding-1">Folding:</a></p></td>
<td><p>From SDSC Spec:</p>
<ul>
<li><p>FoldManager&lt;int64_t&gt; startAddressCoreCorelet_</p></li>
<li><p>first fold is for cores, set as Map fold type</p></li>
</ul>
<ul>
<li><p>second fold is for corelets, set as Const fold type</p></li>
</ul></td>
<td></td>
</tr>
<tr>
<td><ol start="9" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><strong>coordinates</strong>_ (or allocateCoordinates_)</td>
<td><p>Tensor coordinates per dimension</p>
<p>std::map&lt;PrimaryDimTypes, FoldManager&lt;Dtype&gt;&gt; coordinates_</p></td>
<td></td>
<td>Better description of how this field is used is required. An example is provided in the <a href="https://github.com/vswagath1989/torch-spyre/blob/sdsc-bundle-spec/RFCs/0248-SdscBundleSpec/SuperDSC-Bundle.md">SDSC spec</a>, but it does not seem to provide all the details. More on this <a href="#tensor-coordinates">below</a>.</td>
</tr>
<tr>
<td><ol start="10" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">backGapCore_</td>
<td>std::map&lt;PrimaryDimTypes, std::map&lt;int, int&gt;&gt; backGapCore_</td>
<td>See SDSC Bundle Spec</td>
<td>What is the difference between back and front gaps? Illustration with example is required.</td>
</tr>
<tr>
<td><ol start="11" type="1">
<li></li>
</ol></td>
<td style="text-align: center;"><p>indirectAllocType_</p>
<p>relatedIndirectAccessAlloc</p></td>
<td></td>
<td>See SDSC Bundle Spec</td>
<td>More explanation with example desirable</td>
</tr>
</tbody>
</table>

### dscs\_\[idx\]::computeOp\_ (class ComputeOpInfo):

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 24%" />
<col style="width: 23%" />
<col style="width: 22%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>S.No.</strong></th>
<th style="text-align: center;"><strong>Field Name (Mandatory/Optional Indicator)</strong></th>
<th style="text-align: center;"><strong>Purpose / Functionality</strong></th>
<th style="text-align: center;"><strong>How to fill</strong></th>
<th style="text-align: center;"><strong>Additional Comments/Questions</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><ol type="1">
<li></li>
</ol></td>
<td style="text-align: center;">exUnit</td>
<td>The compute engine on which the operation is to be executed. Is an enum specified in SenComponents (@dsc/dscdefn.h). Typically, PE, PT, SFP</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="2" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">opFuncName</td>
<td>Operation to be executed. Name is an enum in OpFuncs (@dsc/dscdefn.h).</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="3" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">attributes_</td>
<td>Specifies attributes such as dataFormat_, fidelity</td>
<td></td>
<td>What is fidelity?</td>
</tr>
<tr>
<td><ol start="4" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">location</td>
<td>Indicates location is loop. Enum LoopNames</td>
<td></td>
<td>Should frontend fill this?</td>
</tr>
<tr>
<td><ol start="5" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">isAtMainLoop, isAtTop, level</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="6" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">coreExclude, coreClExclude</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="7" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">inputLabeledDs</td>
<td>Identifies tensors from labeledDs_ that form inputs to the operation</td>
<td>The number of inputs can be identified from the associated ddl.</td>
<td></td>
</tr>
<tr>
<td><ol start="8" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">outputLabeledDs</td>
<td>Identifies tensors from labeledDs_ that are used to hold outputs of the operation.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="9" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">indirectAccessIndexLabeledDs</td>
<td>Probably related to the recently discussed tensors with indirect addresses.</td>
<td></td>
<td></td>
</tr>
<tr>
<td><ol start="10" type="1">
<li></li>
</ol></td>
<td style="text-align: center;">interimLabeledDs</td>
<td>for partial results and other tensors that live only within the dataflow</td>
<td></td>
<td>Should the frontend need to care about this?</td>
</tr>
</tbody>
</table>

## Folding

Claude Summary:

**Folding** is a compilation technique that generates a **single parameterized SDSC.json** that can execute multiple variants without recompilation. Folding information fields track how dimensions are "folded" (repeated or time-multiplexed) to support:

1\. **Time folding** - Execute operations across multiple iterations

2\. **Space folding** - Repeat computations on different data slices

3\. **Memory paging** - Handle data larger than local memory

4\. **Working set reduction** - Time-multiplex large working sets

5\. **Multi-AIU optimization** - Distribute work across multiple processors

### 

### Questions:

#### Padding:

From SDSC Bundle Spec:

1.  **For window/padded operations, padding information should be added to both N\_ and dataStageParam\_ in sdsc.dscs\_\[0\], capturing information about front/back padding, stride, and related kernel dimension. If a padded dimension is chunked across cores, front/back padding should be set to -1 in core datastage.**

<!-- -->

(a).  In convolution, *i* and *j* dimensions of the input and output tensors can differ depending on padding size. If the input tensor is not padded or padding is less than **stride**, then *i* and *j* dimensions of the output tensor will be smaller than those of the original input tensor. On the other hand, while a padded input tensor will also be larger than the output tensor, its sizes will be larger than the original size specified by the user. Will these dimensions in the two tensors have the same semantics even if their sizes are different? Can a dimension have different sizes in different tensors? Which tensor would the **totalSize\_** field of **paddingSizes** correspond to? Input Tensor only?

(b).  What should be the value of “padding” field in different nodes/blocks: scheduleTree\_::allocate, coordinates\_::coordInfo::\<dim\> (in, out etc) from the following set for the two scenarios above: NOPAD,  LOWERED_PADDED,    PADDED_NOZEROPAD,  PADDED_WZEROPAD,     PADDED_FULLSPAN,     PADDED_FULLSPAN_WUNNEEDED. Will be good to know what each padding type means.

When a dimension is padded due to window/padded operations like convolution, details of padding need to be specified via the following fields.

(i)
```
      "paddingSizes_": {
         "<padded dim>": {
                "padFront_": 0 or 1,
                "padBack_":  0 or 1,
                "totalSize_": total size inclusive of padding, 
                "stride_": <stride applied with the op>,
                "dilation_": <dilation applied with the op>,
                "windowDim_": <window dim> e.g. "ki_"
           }
```
One set of entries need to be added for each dimension that is padded within the paddingSizes_ structure. paddingSizes_ itself would be a sub-field of both dcs[0].N_ and dcs[0].dataStageParam_. 
Regardless of whether padding is applied or not in a convolution operation, totalSize_ will correspond to the input size after padding. E.g. for an image of size 128x128, totalSize_ would be 130 with a padding of 1 and 128 without padding. The dimension's size (as specified in N_, dataStageParam_ per se.) will correspond to the output image size and hence 128 with padding=1 and 126 without padding.
           
(ii) padding_ sub-field in <a href=#scheduletree_-class-scheduletree>scheduleTree</a>, which can take one of the following values:
```
NOPAD
LOWERED_PADDED
PADDED_NOZEROPAD
PADDED_WZEROPAD
PADDED_FULLSPAN
PADDED_FULLSPAN_WUNNEEDED
```
PADDED_NOZEROPAD to be used with conv2d when padding is non-zero.
PADDED_FULLSPAN_WUNNEEDED to be used with conv2d when padding is zero.
   

#### Layout Dimorder (layoutDimorder\_):

2.  **Should the stick dimension(s) always be the rightmost dimension of the layoutDimOrder\_? We see examples of both when this holds and does not hold. As we understand, layoutDimOrder\_ represents the order of dimensions on the device, with the leftmost dimension being the innermost one. Then, since it is the innermost dimension that would be contiguous in memory and fetched in a stick, how is it that the leftmost dimension is generally not specified as the stick dimension in the sdsc’s? The [Tiled Tensors RFC](https://github.com/torch-spyre/torch-spyre/blob/main/RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md) also alludes to the innermost dimension being the stick dimension in [Section Padded Tensors](https://github.com/torch-spyre/torch-spyre/blob/main/RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md#padded-tensors).**

**Explanation:**

PrimaryDsInfo -\> the layoutDimOrder\_ contains all the dimensions related to a labeledDs (there is no ordering).

The actual layout (beyond a stick) of the tensor is specified (from inner to outer) in **allocate** node of **scheduleTree**. For example, in a 2D tensor with dimensions out=512, mb=4 elements and out=64 elements in the stick, the tensor will be \[stick-out=64\]\[layout-mb=4\]\[layout-out=8\]. So the layoutDimOrder\_ should be \[mb\]\[out\] and the size can be \[-1, -1\] i.e. all elements outside of stick.

When number of elements == stick size, that dimension still has to be added to the layout (with size=1). E.g., 2D tensor with out=64 mb=4 elements with out=64 elements in the stick, then  \[stick-out=64\]\[layout-mb=4\]\[layout-out=1\]

**Follow-on Question:**

Where should size be specified?

#### Tensor Coordinates

3.  **The example in the SDSC Bundle Spec is not clear. Does the coordinate sequence 0, 1, 2, 3, 64, 65, 66, 67, 4, 5, 6, 7, 68, 69, 70, 71 provided denote the ordering in memory. How should it be handled depending on the position of the dimension in the layout? In general, the structure to fill is of the form:**

```
DIM (e.g."mb"):
{
  "spatial": 3,
  "temporal": 0,
  "elemArr": 1,
  "padding": "nopad",
  "folds": {
  "dim_prop_func": [
      { "Affine": {"alpha_":8,"beta_":0} },
      { "Affine": {"alpha_":0,"beta_":0}},
      { "Affine": {"alpha_":0,"beta_":0}},
      { "Affine": {"alpha_":1,"beta_":0}}
  ],

  "dim_prop_attr": [
      { "factor_":1,"label\_":"core_fold"},
      { "factor_":1,"label\_":"corelet_fold"},
      { "factor_":1,"label\_":"row_fold"},
      { "factor_":8,"label\_":"elem_arr_0"}
    ]
  }
}

```

coordinates_::coordInfo has one entry per dimension in scheduleTree’s allocate node.

Stick dimension has an additional element in both **dim_prop_func** and **dim_prop_attr** for “**elem_arr_1**”.

Need details on how to determine the values for alpha_, beta_ and factor_ for each type of fold. Its purpose appears to be to determine the offset of any element along the specified dimension of a tensor. A non-trivial and complete example illustrating the determination of values will be helpful.

Some explanation is provided in <a href=#folding-1>Sec. Folding</a> below.

#### scale_:

4.  **Can we have some examples where scale_ is -1? This question is prompted by the following:**

In matmul (and batch matmul), each output element is obtained by reducing a (row, column) combination of the input tensors. Using the conventional names of x, mb, in, out for the dimensions, does this mean **in** dimension of the first and second tensors should have a scale of -1, since they are involved in a reduction? Or because these don't vanish in the input tensors should they remain as 1? What should be the scale vector for the output tensor?

#### Folding:

**The main folding related data structures are as follows:**

```
class FoldManger {
  FoldFunction<Dtype>* parent_func_ = nullptr;
  fm_dim_prop dim_prop_;
}

using fm_dim_prop = std::vector<std::pair<const FoldDimProp*, BaseFuncType>>;

class FoldDimProp {
  unit32_t factor_;
  std::string label_;
}

enum class BaseFuncType {
  Constant = 0,
  Map = 1,
  Affine = 2,
  WkSplit = 3,
  Unknown = 4
};
```

In the SDSC, these structures are transformed to json of following format:

```
"dim_prop_func": [
  {
    "Affine": {"alpha_":1,"beta_":0}
  }
],

"dim_prop_attr": [
  {"factor_":1,"label_":"time"}  
],

"data_": {"[0]":"0"}
```

**This generic format is used for a few fields such as sdsc.sdscFolds\_, scheduleTree.startAddressCoreCorelet, scheduleTree.coordinates\_.coordInfo.\<dimname\>.folds. What does the generic structure represent and how should it be filled in each case?**

**Explanation:**

##### scheduleTree.startAddressCoreCorelet\_:

This field lists the starting memory addresses for each core, corelet, and timesteps for each of the tensors used. Hence, it is part of allocate node of schedule tree. In this case, the dim_prop_func field is almost always set as follows.

```
"dim_prop_func": [
   { "Map": {} },
   {  "Const": {} },
]
```

In dim_prop_attr below, the first entry corresponds to core, the second to corelet, and the final to addresses for different times. Only start addresses per core are indicated from the front end. The other two are set to const. dim_prop_attr field has to indicate the splitting factor for cores and corelets, while it is 1 for time from the front end. Example below:

```
"dim_prop_attr": [
   {"factor_":20,"label_":"core"},
   {"factor_":1,"label_":"corelet"},
   {"factor_":1,"label_":"time"}
],
```

Now, the Map{} entry in the first structure indicates that the start addresses for the tensor for various cores are provided in the “data\_” map. E.g.:

```
 "data_": {"[0, 0, 0]":"0","[1, 0, 0]":"0","[2, 0, 0]":"0","[3, 0, 0]":"0","[4, 0, 0]":"0","[5, 0, 0]":"128","[6, 0, 0]":"128","[7, 0, 0]":"128","[8, 0, 0]":"128","[9, 0, 0]":"128","[10, 0, 0]":"256","[11, 0, 0]":"256","[12, 0, 0]":"256","[13, 0, 0]":"256","[14, 0, 0]":"256","[15, 0, 0]":"384","[16, 0, 0]":"384","[17, 0, 0]":"384","[18, 0, 0]":"384","[19, 0, 0]":"384"}
```

Rules for determining the addresses depend on how a tensor’s dimensions are split.

The above example corresponds to a case when the first tensor has a 4-way split. Each slice is used by 5 cores and hence the start address remains the same for 4 of the 20 cores. Each slice is just 128 bytes long, and hence the start addresses of successive slices are 128 bytes apart.

##### scheduleTree.allocate Node’s coordinates\_.coordInfo.\<dim name\>.folds:

Under scheduleTree, the **dim_prop_func** and **dim_prop_attr** sub-structures are used to describe how each dimension of a tensor is progressively split across cores, corelets, rows, and the final leaf entities, referred to using dim_prop_attr sub-structure labels core_fold, corelet_fold, row_fold, elem_arr_0, and elem_arr_1, respectively. Consider a tensor with dimensions *x*, *in*, and *out* with sizes given by \[4, 2880, 2880\]. The *x* dimension has a *4*-way split and the *out* dimension, a 5-way split while the *in* dimension is not split. The total number of splits (tensor sub-blocks) is hence 20, with each split assigned to a core, for 20 cores in all. scheduleTree’s coordinates\_ provides details of these splits.

Each allocate node of scheduleTree\_ (corresponding to a tensor) contains a coordinates\_ sub-field, with the following schema:

```
"coordinates_": {
"coordInfo": {
  <dim_name>: {
    "spatial": 3,
    "temporal": 0,
    "elemArr": 1 or 2,
    "padding": <padding type>,
    "folds": {
      "dim_prop_func": [
      { "Affine": {"alpha_":<number of cores spanned by each core-wise split>, "beta_":0} },
      { "Affine": {"alpha_":1, "beta_":0} },
      { "Affine": {"alpha_":1, "beta_":0} },
      { "Affine": {"alpha_":number of lower-most splits, typically 1, "beta_":0} },
  ],
    
  "dim_prop_attr": [
      {"factor_": <number of core-wise splits >, "label_":"core_fold"},
      {"factor_":<number of corelet splits>, "label_":"corelet_fold"},
      {"factor_":<number of row splits>, "label_":"row_fold"},
      {"factor_":<number of elements in the lower-most slice>, "label_":"elem_arr_0"}
  ]
 }
},
```

**coordInfo.spatial** is typically 3, indicating that there are 3 spatial splits, along cores, corelets, and rows.

**coordInfo.temporal** is set to 0 from the front end.

coordInfo.elemArr is 1 for non-stick dimensions and 2 for stick dimensions. It is 2 for stick dimensions as the elements along the stick dimension are broken down into sticks. While a stick is at the lowest level, the number of sticks in a dimension’s slice is at the next higher level.

For a stick dimension, **dim_prop_attr** includes an entry with label set to **elem_arr_1**, whose corresponding factor field would denote the number of sticks in each slice corresponding to a core, corelet, row combination, that is, the slice size/# of elements per stick. factor\_ corresponding to elem_arr_0 would indicate the number of elements per stick.

For our example tensor, folds field for various dimensions would be as follows:

For dimension *x*:

```
"folds": {
"dim_prop_func": [
    { "Affine": {"alpha_":1,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":1,"beta_":0} }
  ],

"dim_prop_attr": [
    {"factor_":4,"label_":"core_fold"},
    {"factor_":1,"label_":"corelet_fold"},
    {"factor_":1,"label_":"row_fold"},
    {"factor_":1,"label_":"elem_arr_0"}
  ]
}
```

For dimension out:

```
"folds": {
  "dim_prop_func": [
    { "Affine": {"alpha_":576,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":64,"beta_":0} },
    { "Affine": {"alpha_":1,"beta_":0} }
  ],

  "dim_prop_attr": [
    {"factor_":5,"label_":"core_fold"},
    {"factor_":1,"label_":"corelet_fold"},
    {"factor_":1,"label_":"row_fold"},
    {"factor_":9,"label_":"elem_arr_1"},
    {"factor_":64,"label_":"elem_arr_0"}
  ]
}
```

Since out dimension is a stick dimension, it also includes details for elem_arr_1.

For dimension *in*:

```
"folds": {
  "dim_prop_func": [
    { "Affine": {"alpha_":2880,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":0,"beta_":0} },
    { "Affine": {"alpha_":1,"beta_":0} }
  ],

  "dim_prop_attr": [
    {"factor_":1,"label_":"core_fold"},
    {"factor_":1,"label_":"corelet_fold"},
    {"factor_":1,"label_":"row_fold"},
    {"factor_":2880,"label_":"elem_arr_0"}
  ]
}
```

