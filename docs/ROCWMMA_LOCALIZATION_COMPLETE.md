# rocWMMA.hpp Localization Complete

## Summary

The main rocWMMA header file has been successfully localized using Doxygen's `\~` language switch feature. All public API documentation is now available in three languages:

- **English** (original)
- **Chinese (Simplified)** (简体中文)
- **Japanese** (日本語)

## Localized Components

### 1. Main Page Documentation
- Overview of rocWMMA library
- Architecture description
- Feature highlights
- Usage information

### 2. Data Structure Tags
- `row_major` - Row-major layout tag
- `col_major` - Column-major layout tag
- `matrix_a` - Matrix A context tag
- `matrix_b` - Matrix B context tag
- `accumulator` - Accumulator context tag
- `layout_t` - Runtime layout enumeration

### 3. Fragment Schedulers
- `default_schedule` - Default independent scheduling
- `coop_row_major_2d` - Cooperative row-major 2D scheduling
- `coop_col_major_2d` - Cooperative column-major 2D scheduling
- `coop_row_slice_2d` - Cooperative row-slice 2D scheduling
- `coop_col_slice_2d` - Cooperative column-slice 2D scheduling
- `single` - Single wave scheduling

### 4. Fragment Class
Complete documentation for the `fragment` template class including:
- Class overview and purpose
- Template parameters
- Usage notes

### 5. Core API Functions

#### fill_fragment
Fills a fragment with a constant value
- **English**: Fills the entire fragment with the desired value.
- **Chinese**: 用指定值填充整个片段。
- **Japanese**: フラグメント全体を指定された値で埋める。

#### load_matrix_sync (2 overloads)
Loads fragment data from memory
- Basic version with automatic layout detection
- Advanced version with manual layout specification

#### store_matrix_sync (2 overloads)
Stores fragment data to memory
- Basic version with automatic layout detection
- Advanced version with manual layout specification

#### mma_sync
Performs matrix multiply-accumulate operation (D = A * B + C)
- **English**: Performs the Multiply-Accumulate operation on the fragments
- **Chinese**: 对片段A、B、C和D执行乘累加操作
- **Japanese**: フラグメントA、B、C、Dに対して乗算累積演算を実行する

#### synchronize_workgroup
Synchronizes all wavefronts in a workgroup
- **English**: Synchronization point for all wavefronts in a workgroup
- **Chinese**: 工作组中所有波前的同步点
- **Japanese**: ワークグループ内のすべてのウェーブフロントの同期ポイント

## Building Documentation in Different Languages

To generate documentation in a specific language, set the `OUTPUT_LANGUAGE` parameter in your Doxyfile:

### English (Default)
```bash
OUTPUT_LANGUAGE = English
doxygen Doxyfile
```

### Chinese
```bash
OUTPUT_LANGUAGE = Chinese
doxygen Doxyfile
```

### Japanese
```bash
OUTPUT_LANGUAGE = Japanese
doxygen Doxyfile
```

## Integration with Sphinx + Breathe

If you're using Sphinx with the Breathe extension to integrate Doxygen documentation:

1. Generate Doxygen XML for each language:
```bash
# English
sed -i 's/OUTPUT_LANGUAGE.*/OUTPUT_LANGUAGE = English/' docs/doxygen/Doxyfile
doxygen docs/doxygen/Doxyfile
mv docs/doxygen/xml docs/doxygen/xml_en

# Chinese
sed -i 's/OUTPUT_LANGUAGE.*/OUTPUT_LANGUAGE = Chinese/' docs/doxygen/Doxyfile
doxygen docs/doxygen/Doxyfile
mv docs/doxygen/xml docs/doxygen/xml_zh

# Japanese
sed -i 's/OUTPUT_LANGUAGE.*/OUTPUT_LANGUAGE = Japanese/' docs/doxygen/Doxyfile
doxygen docs/doxygen/Doxyfile
mv docs/doxygen/xml docs/doxygen/xml_ja
```

2. Update `conf.py` to point to the appropriate XML directory based on language:
```python
# For English
breathe_projects = {"rocwmma": "doxygen/xml_en"}

# For Chinese
breathe_projects = {"rocwmma": "doxygen/xml_zh"}

# For Japanese
breathe_projects = {"rocwmma": "doxygen/xml_ja"}
```

3. Build Sphinx documentation for each language:
```bash
# English
sphinx-build -b html -D breathe_projects.rocwmma=doxygen/xml_en docs _build/html/en

# Chinese
sphinx-build -b html -D breathe_projects.rocwmma=doxygen/xml_zh -D language=zh_CN docs _build/html/zh

# Japanese
sphinx-build -b html -D breathe_projects.rocwmma=doxygen/xml_ja -D language=ja docs _build/html/ja
```

## Technical Terminology Glossary

Key technical terms and their translations:

| English | Chinese (简体中文) | Japanese (日本語) |
|---------|-------------------|-------------------|
| fragment | 片段 | フラグメント |
| matrix | 矩阵 | 行列 |
| accumulator | 累加器 | アキュムレータ |
| wavefront | 波前 | ウェーブフロント |
| row major | 行主序 | 行優先 |
| column major | 列主序 | 列優先 |
| thread block | 线程块 | スレッドブロック |
| scheduler | 调度器 | スケジューラ |
| multiply-accumulate | 乘累加 | 乗算累積 |
| workgroup | 工作组 | ワークグループ |
| leading dimension | 主维度 | 主次元 |
| cooperative | 协作 | 協調 |
| synchronization | 同步 | 同期 |
| data layout | 数据布局 | データレイアウト |

## Files Modified

- `library/include/rocwmma/rocwmma.hpp` - Main API header with multi-language documentation

## Verification

To verify the localization:

1. **Build HTML documentation** in each language and visually inspect
2. **Check XML output** to ensure language tags are properly processed
3. **Test search functionality** with non-ASCII characters (Chinese/Japanese)
4. **Verify cross-references** work correctly in all languages

## Example Code with Localized Documentation

When viewing the generated documentation in Chinese, users will see:

```cpp
template <typename FragT, typename DataT>
void fill_fragment(FragT& frag, DataT value);
```

**中文文档:**
用指定值填充整个片段。

**参数:**
- `frag` - MatrixT类型的片段，包含其关联的块大小、数据类型和布局
- `value` - DataT类型的填充值

**模板参数:**
- `FragT` - 不透明片段类型
- `DataT` - 数据类型

## Benefits

1. **Single Source of Truth**: All translations in one file, synchronized with code
2. **Build-time Selection**: Choose output language at documentation build time
3. **No Runtime Overhead**: Language selection happens during documentation generation
4. **Maintainability**: Updates to code documentation automatically update all languages
5. **International Accessibility**: Users can read documentation in their preferred language

## Limitations

1. **One Language Per Build**: Doxygen can only generate documentation in one language at a time
2. **Manual Translation**: All translations must be maintained manually in source code
3. **Build Complexity**: Requires separate build processes for each language
4. **No Fallback**: Missing translations won't fall back to English

## Best Practices

1. **Keep translations synchronized**: When updating English docs, update all translations
2. **Use consistent terminology**: Follow the glossary for technical terms
3. **Preserve formatting**: Maintain the same structure across all languages
4. **Don't translate code**: Only translate natural language descriptions
5. **Test all languages**: Build and verify documentation in all supported languages

## Future Enhancements

Potential improvements for the localization system:

1. Add more languages (Korean, French, German, etc.)
2. Create automated translation verification scripts
3. Implement translation memory to ensure consistency
4. Develop CI/CD pipeline for multi-language documentation builds
5. Add language-specific examples in code samples

## References

- [Doxygen Language Support](https://www.doxygen.nl/manual/langhowto.html)
- [Doxygen Special Commands](https://www.doxygen.nl/manual/commands.html)
- [Example Localization Files](LOCALIZATION_EXAMPLE.md)

---

**Date Localized**: 2026-06-21  
**Languages**: English, Chinese (Simplified), Japanese  
**Total Documentation Blocks Localized**: 20+  
**Status**: Complete ✓
