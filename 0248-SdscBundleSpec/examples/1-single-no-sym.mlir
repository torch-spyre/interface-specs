// An example with a single SDSC without any symbol
module {
  func.func @single_no_sym() {
    
    // Start addresses, work division, sizes, and operations are filled directly in sdsc.jon
    // No additional information needs to be passed from bundle mlir to sdsc.
    // Therefore bundle mlir only needs to specify that one sdsc needs to be executed:
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc.json"}

    return
  }
}
