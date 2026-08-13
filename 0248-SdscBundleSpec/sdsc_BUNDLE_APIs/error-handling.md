# Error Handling

## Common Errors

### Invalid Symbol Count

**Error:**

```
Error: Number of operands (3) does not match symbol_ids count (2)
```

**Solution:** Ensure operand count matches `symbol_ids` array length

### Missing SDSC File

**Error:**

```
Error: Cannot find sdsc.json at specified path
```

**Solution:** Verify `sdsc_filename` path is correct and relative to MLIR file

### Invalid Stick Layout

**Error:**

```
Error: Stick layout violates constraints for operation type
```

**Solution:** Review stick constraints for operation category (see [Stick Layout Constraints](08-mlir-stick-layout-constraints.md))

### Symbolic Dimension Division

**Error:**

```
Error: Number of slices (3) does not divide granularity (8)
```

**Solution:** Ensure slice count is divisor of granularity

## Validation Checklist

- ☐ All `symbol_ids` have corresponding operands
- ☐ SDSC filenames are valid and accessible
- ☐ Stick layouts meet operation constraints
- ☐ Symbolic dimensions properly configured
- ☐ Work division matches tensor allocation
- ☐ Data formats are consistent
- ☐ Constants are properly defined

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Best Practices →](best-practices.md) |
|:--|:--:|--:|
