// In this example we want to demonstrate a bundle.mlir where symbols are passed into
// SDSCs. They are used to specify start addresses for every core for each tensor.
// These symbols are "fake" because they are constant in the mlir but they are still
// passed to the sdsc as symbols for illustration.
// NOTE: having actual symbols in mlir will be supported through the next revision of the spec

module {
  func.func @single_fake_sym() {
    %inp_core0_start_address = arith.constant 1024 : index
    %inp_core1_start_address = arith.constant 1152 : index
    %outp_core0_start_address = arith.constant 4096 : index
    %outp_core1_start_address = arith.constant 4224 : index
    
    // symbol ids -1, -2,... must be used in the allocation nodes of sdscGelu.json to specify the start
    // address for every slice and every tensor. The corresponding %inp_core0_start_address,... (in the same order) 
    // give the actual value to substitute in the program
    sdscbundle.sdsc_execute (%inp_core0_start_address, %inp_core1_start_address, %outp_core0_start_address, %outp_core1_start_address)
           {sdsc_filename="sdscGelu.json", symbol_ids=[-1, -2, -3, -4]}

    // example of symbolic start addresses in allocate node for inp tensor in sdscGelu.json:
    // "startAddressCoreCorelet_" :  {
    //   "dim_prop_func" : [
    //     { "Map" : {} },
    //     { "Const" : {} },
    //   ],
    //   "dim_prop_attr" : [
    //     { "factor_" : 2, "label_" : "core" },
    //     { "factor_" : 2, "label_" : "corelet" },
    //   ],
    //   "data_" : {
    //     "[0, 0]" :"-1",
    //     "[1, 0]" :"-2"
    //   }
    // },
    // "isStartAddrSymbolic_" : 1,


    return
  }
}
