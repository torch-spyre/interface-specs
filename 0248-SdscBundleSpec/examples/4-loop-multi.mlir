// An example with multiple SDSCs in a loop. Start addresses change at every invokation of a SDSC based on the 
// loop iteration but the base address of the tensors are constant and known.

#B_start_address_map = affine_map<(d0, core)[base] -> (base + 1024*d0 + 256*core)>
#C_start_address_map = affine_map<(d0, core)[base] -> (base + 256*d0 + 512*core)>

module {
  func.func @loop_multi() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop_bound = arith.constant 8 : index

    %core0 = arith.constant 0 : index
    %core1 = arith.constant 1 : index

    %A_base_address = arith.constant 1024 : index
    %B_base_address = arith.constant 12288 : index
    %C_base_address = arith.constant 17888 : index

    scf.for %i = %c0 to %loop_bound step %c1 {  // no loop carried variables

      // Addresses can only be generated with affine expressions, using constants and loop iterators.
      // Use of other dialects to generate symbol values (e.g. arith) will be enhanced in the future.
      // We don't support scf.if, neither for conditional symbol value definitions nor for conditional execution of SDSCs.

      // Affine expression to calculate address as base_address + 128 * loop_iterator.
      // If the dimension split across cores is not present in tensor A, a single start address is sufficient for all cores
      %A_start_address = affine.apply affine_map<(d0)[base] -> (base + 128 * d0)> (%i)[%A_base_address]

      // Affine expression to calculate address as base_address + 1024 * loop_iterator + 256 * core_slice
      %B_start_address_core0 = affine.apply #B_start_address_map (%i, %core0)[%B_base_address]
      %B_start_address_core1 = affine.apply #B_start_address_map (%i, %core1)[%B_base_address]
      
      sdscbundle.sdsc_execute (%A_start_address, %B_start_address_core0, %B_start_address_core1) {sdsc_filename="sdscA.json", symbol_ids=[-1, -2, -3]}
      
      // Affine expression to calculate address as base_address + 256 * loop_iterator + 512 * core_slice
      %C_start_address_core0 = affine.apply #C_start_address_map (%i, %core0)[%C_base_address]
      %C_start_address_core1 = affine.apply #C_start_address_map (%i, %core1)[%C_base_address]
      
      sdscbundle.sdsc_execute (%B_start_address_core0, %B_start_address_core1, %C_start_address_core0, %C_start_address_core1) {sdsc_filename="sdscB.json", symbol_ids=[-4, -5, -6, -7]}

    }

    return
  }
}
