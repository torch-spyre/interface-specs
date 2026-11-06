// An example with a series of SDSCs without any symbol
module {
  func.func @softmax_no_sym() {
    
    // Softmax can be realized by decomposing it in six operations that can be specified in sequence
    // by a single SuperDSC-Bundle. In this example we assume shapes and start addresses are fixed, so 
    // no symbol values are passed to the SDSCs.

    // y = sofmax(x, dim=1)

    // x_max = max(x, dim=1)
    sdscbundle.sdsc_execute () {sdsc_filename="sdscMax.json"}
    // x_sub = x - x_max
    sdscbundle.sdsc_execute () {sdsc_filename="sdscSub.json"}
    // x_exp = exp(x_sub)
    sdscbundle.sdsc_execute () {sdsc_filename="sdscExp.json"}
    // x_sum = sum(x_exp, dim=1)
    sdscbundle.sdsc_execute () {sdsc_filename="sdscSum.json"}
    // x_recpr = 1/x_sum
    sdscbundle.sdsc_execute () {sdsc_filename="sdscReciprocal.json"}
    // y = x_sub * x_recpr
    sdscbundle.sdsc_execute () {sdsc_filename="sdscMul.json"}

    return
  }
}
