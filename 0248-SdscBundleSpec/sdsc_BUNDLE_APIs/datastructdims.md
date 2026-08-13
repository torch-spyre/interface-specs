# DataStructDims

Data structure dimensions with name and dimension values.

## Structure

```json
{
  "name_": "<string>",
  "<dim_name>_": <int>,
  ...
}
```

## Fields

- **`name_`**: (optional) Dimension set name
- **`<dim_name>_`**: (pattern) Direct dimension properties (e.g., `mb_`, `out_`, `in_`)
  - Pattern: `^(?!name_$)[A-Za-z_][A-Za-z0-9_]*_$` (excludes `name_`)
  - Type: integer >= 1

## Example

```json
{
  "name_": "core",
  "mb_": 32,
  "out_": 128,
  "in_": 64
}
```

---

[← Back to Table of Contents](README.md)
