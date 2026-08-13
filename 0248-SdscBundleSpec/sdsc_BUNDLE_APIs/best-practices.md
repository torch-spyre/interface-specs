# Best Practices

## Performance Optimization

1. **Minimize Data Movement**

   - Place frequently accessed tensors in LX scratchpad
   - Use HBM for large, infrequently accessed data

2. **Balance Work Division**

   - Distribute work evenly across cores
   - Consider memory bandwidth constraints

3. **Optimize Stick Layouts**

   - Align stick dimensions with computation patterns
   - Minimize padding requirements

## Code Organization

1. **Naming Conventions**

   - Use descriptive SDSC filenames: `sdscGelu.json`, `sdscMatmul.json`
   - Use meaningful tensor labels: `"input"`, `"weight"`, `"output"`

2. **Symbol Management**

   - Use sequential negative IDs: `-1`, `-2`, `-3`
   - Document symbol purpose in comments

3. **Modularity**

   - Create reusable SDSC files for common operations
   - Use separate SDSCs for distinct operations

## Debugging

1. **Incremental Development**

   - Start with single operation, no symbols
   - Add complexity gradually

2. **Validation**

   - Verify SDSC JSON structure before integration
   - Test with fixed addresses before adding symbols

3. **Documentation**

   - Comment complex affine maps
   - Document work division strategy

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Reference →](reference.md) |
|:--|:--:|--:|
